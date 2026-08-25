"""
AIPAF — AI Project Assessment Framework
agent.py — Orchestration layer (LLM + ScoringEngine)

Responsabilita':
  1. Usa l'LLM per:
     - raccogliere dati intake in linguaggio naturale
     - interpretare le risposte e strutturarle in CriterionResponse
     - generare il testo narrativo del report finale
  2. Chiama ScoringEngine per il calcolo deterministico degli score
  3. Non contiene logica di scoring: quella appartiene a engine.py

Principi di governance (v0.2.0):
  - Le chiamate strutturate (intake, criteri) sono STATELESS: ogni chiamata
    parte da una conversazione vuota, così la valutazione di un criterio non
    è contaminata da quelle precedenti. Lo storico serve solo alle narrative.
  - I fallimenti dell'LLM sono RUMOROSI: JSON malformato o valori fuori dominio
    sollevano LLMOutputError dopo un retry. Nessun default silenzioso.

Autore: Lorenzo Lombardi
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional, TYPE_CHECKING

from .engine import FrameworkConfig, ScoringEngine
from .llm.base import LLMProvider, LLMRole, Message
from .models import (
    REMEDIATION_THRESHOLD,
    AssessmentSession,
    ConfidenceLevel,
    CriterionResponse,
    DimensionConfig,
    EUAIActLevel,
    ProjectIntake,
    Sector,
)

if TYPE_CHECKING:
    from .rag.retriever import RAGRetriever

log = logging.getLogger("aipaf.agent")

STRUCTURED_RETRIES = 1   # tentativi aggiuntivi dopo il primo fallimento


class LLMOutputError(RuntimeError):
    """L'LLM ha prodotto un output non utilizzabile (JSON malformato, valori fuori dominio)."""


# ---------------------------------------------------------------------------
# System prompt base per l'agent
# ---------------------------------------------------------------------------

_BASE_SYSTEM_PROMPT = """\
Sei un esperto di AI Governance che conduce assessment strutturati secondo \
il framework AIPAF (AI Project Assessment Framework).

Il framework valuta i progetti AI su 6 dimensioni:
  D1 — Strategic Alignment & Business Value
  D2 — Risk Classification & Regulatory Compliance
  D3 — Data Readiness & Governance
  D4 — Technical Feasibility & Architecture
  D5 — Ethical & Responsible AI
  D6 — Operational Readiness & Lifecycle Management

Ogni criterio viene valutato su scala 0-3:
  3 = Excellent  — best practice, evidenze documentate
  2 = Adequate   — gap identificati, remediation plan
  1 = Insufficient — gap significativi, intervento necessario
  0 = Critical   — non soddisfatto, stop obbligatorio

La tua confidence nell'evidenza fornita:
  H = High   — evidenza documentata verificabile  (moltiplicatore 1.0)
  M = Medium — auto-dichiarazione senza prove     (moltiplicatore 0.8)
  L = Low    — stima o assenza di dato            (score azzerato, data needed)

Quando interpreti una risposta devi sempre:
1. Identificare lo score numerico (0-3) piu' appropriato
2. Valutare la confidence in base alla qualita' dell'evidenza descritta
3. Estrarre l'evidenza rilevante dal testo
4. Rispondere SOLO con JSON valido quando richiesto, senza testo aggiuntivo
"""

_JSON_ONLY = "\n\nRispondi ESCLUSIVAMENTE con JSON valido. Nessun testo prima o dopo."

_RETRY_SUFFIX = (
    "\n\nATTENZIONE: la risposta precedente non era JSON valido o conteneva valori "
    "non ammessi. Restituisci SOLO l'oggetto JSON richiesto, senza commenti, "
    "senza backtick, con i valori esattamente tra quelli ammessi."
)


# ---------------------------------------------------------------------------
# AssessmentAgent
# ---------------------------------------------------------------------------

class AssessmentAgent:
    """
    Orchestratore principale dell'assessment AIPAF.

    Flusso:
      1. collect_intake()                — raccoglie dati progetto
      2. interpret_criterion_response()  — per ciascun criterio
      3. finalize()                      — calcola score (engine)
      4. generate_executive_summary() / generate_remediation_plan() — narrative
    """

    def __init__(
        self,
        provider:  LLMProvider,
        config:    FrameworkConfig,
        engine:    ScoringEngine,
        *,
        retriever: Optional["RAGRetriever"] = None,
        verbose:   bool = False,
    ):
        self.provider  = provider
        self.config    = config
        self.engine    = engine
        self.retriever = retriever   # None → RAG disabilitato
        self.verbose   = verbose
        self._history: list[Message] = []   # solo per le narrative

    # ------------------------------------------------------------------
    # Conversazione
    # ------------------------------------------------------------------

    def _complete(self, messages: list[Message], *, json_mode: bool = False) -> str:
        system = _BASE_SYSTEM_PROMPT + (_JSON_ONLY if json_mode else "")
        response = self.provider.complete(messages=messages, system=system, temperature=0.1)
        if self.verbose:
            print(f"\n[{self.provider.provider_name}/{self.provider.model_name}]"
                  f"  tokens: {response.total_tokens}")
        return response.content

    def _chat(self, user_message: str) -> str:
        """Chiamata narrativa con storico (executive summary, remediation)."""
        self._history.append(Message(LLMRole.USER, user_message))
        content = self._complete(self._history)
        self._history.append(Message(LLMRole.ASSISTANT, content))
        return content

    def _structured(self, prompt: str, validate) -> dict:
        """
        Chiamata strutturata STATELESS con retry.

        `validate(data) -> dict` normalizza e verifica il JSON; solleva
        LLMOutputError se i valori sono fuori dominio. Dopo STRUCTURED_RETRIES
        tentativi aggiuntivi l'errore viene propagato.
        """
        last_error: Optional[Exception] = None
        current_prompt = prompt
        for attempt in range(1 + STRUCTURED_RETRIES):
            raw = self._complete([Message(LLMRole.USER, current_prompt)], json_mode=True)
            try:
                return validate(_parse_json(raw))
            except LLMOutputError as e:
                last_error = e
                log.warning("Output LLM non valido (tentativo %d/%d): %s",
                            attempt + 1, 1 + STRUCTURED_RETRIES, e)
                current_prompt = prompt + _RETRY_SUFFIX
        raise LLMOutputError(
            f"Output LLM non utilizzabile dopo {1 + STRUCTURED_RETRIES} tentativi "
            f"({self.provider.provider_name}/{self.provider.model_name}): {last_error}"
        )

    def reset_history(self) -> None:
        """Azzera lo storico narrativo."""
        self._history = []

    # ------------------------------------------------------------------
    # Fase 1: Intake
    # ------------------------------------------------------------------

    def collect_intake(self, session: AssessmentSession, raw_description: str) -> AssessmentSession:
        """
        Estrae i dati di intake da una descrizione in linguaggio naturale.
        Solleva LLMOutputError se l'LLM non produce un intake valido:
        settore e livello EU AI Act condizionano pesi e flag normativi,
        quindi non possono essere dedotti per default.
        """
        sectors = ", ".join(f'"{s.value}"' for s in Sector)
        levels  = ", ".join(f'"{l.value}"' for l in EUAIActLevel)
        prompt = f"""
Analizza questa descrizione di progetto AI ed estrai i dati di intake strutturati.

DESCRIZIONE PROGETTO:
{raw_description}

Restituisci un JSON con questa struttura esatta:
{{
  "project_name": "nome del progetto",
  "use_case_description": "descrizione sintetica dello use case",
  "project_owner": "nome del responsabile se menzionato, altrimenti 'Da definire'",
  "sector": uno tra [{sectors}],
  "ai_system_type": "tipo sistema AI (classificatore, chatbot, recommender, ecc.)",
  "eu_ai_act_level": uno tra [{levels}],
  "project_stage": "stadio attuale (PoC, Pilota, Produzione, ecc.)",
  "user_count_estimate": <intero, oppure null se non deducibile>,
  "vulnerable_subjects": true/false,
  "personal_data_treated": true/false,
  "sensitive_data_treated": true/false
}}

Per eu_ai_act_level usa questa logica:
- UNACCEPTABLE: pratiche vietate (Art. 5)
- HIGH_RISK: infrastrutture critiche, healthcare, employment, justice, biometria
- LIMITED_RISK: chatbot, deepfake, sistemi conversazionali
- MINIMAL_RISK: tutto il resto
"""
        data = self._structured(prompt, _validate_intake)

        session.intake = ProjectIntake(
            project_name          = data["project_name"],
            use_case_description  = data["use_case_description"],
            project_owner         = data["project_owner"],
            sector                = data["sector"],
            ai_system_type        = data["ai_system_type"],
            eu_ai_act_level       = data["eu_ai_act_level"],
            project_stage         = data["project_stage"],
            user_count_estimate   = data["user_count_estimate"],
            vulnerable_subjects   = data["vulnerable_subjects"],
            personal_data_treated = data["personal_data_treated"],
            sensitive_data_treated= data["sensitive_data_treated"],
            assessment_date       = _today(),
        )
        return session

    # ------------------------------------------------------------------
    # Fase 2: Assessment per dimensione
    # ------------------------------------------------------------------

    def build_criterion_question(
        self,
        dim_cfg:     DimensionConfig,
        criterion_id: str,
        session:     AssessmentSession,
    ) -> str:
        """Domanda per un criterio, contestualizzata al progetto."""
        crit = next((c for c in dim_cfg.criteria if c.id == criterion_id), None)
        if crit is None:
            return f"Descrivi lo stato del criterio {criterion_id}."

        project_ctx = ""
        if session.intake:
            project_ctx = (
                f"\nProgetto: {session.intake.project_name}"
                f"\nSettore: {session.intake.sector.value}"
                f"\nStadio: {session.intake.project_stage}"
            )

        return f"""
{project_ctx}

Dimensione: {dim_cfg.name}
Criterio [{crit.id}]: {crit.name}
Descrizione: {crit.description}
Fonte: {crit.source}

Scala di valutazione:
  3 (Excellent): {crit.scoring_guide.s3}
  2 (Adequate):  {crit.scoring_guide.s2}
  1 (Insufficient): {crit.scoring_guide.s1}
  0 (Critical):  {crit.scoring_guide.s0}

Come valuti questo criterio per il progetto? Descrivi la situazione attuale,
le evidenze disponibili e il livello di maturita' riscontrato.
"""

    def interpret_criterion_response(
        self,
        criterion_id: str,
        user_answer:  str,
        dim_cfg:      DimensionConfig,
    ) -> CriterionResponse:
        """
        Interpreta la risposta in linguaggio naturale e la struttura in un
        CriterionResponse. Chiamata stateless. Solleva LLMOutputError se
        l'output non è utilizzabile: sta al chiamante decidere come
        registrare il criterio (la CLI registra confidence L con warning).
        """
        crit = next((c for c in dim_cfg.criteria if c.id == criterion_id), None)
        scoring_context = ""
        if crit:
            scoring_context = f"""
Scala per questo criterio:
  3: {crit.scoring_guide.s3}
  2: {crit.scoring_guide.s2}
  1: {crit.scoring_guide.s1}
  0: {crit.scoring_guide.s0}"""

        rag_context = ""
        if self.retriever and crit:
            rag_context = self.retriever.context_for_criterion(
                criterion_id=crit.id,
                criterion_name=crit.name,
                criterion_description=crit.description,
                n_results=3,
            )
            if rag_context:
                rag_context = f"\n{rag_context}\n"

        prompt = f"""
Criterio: {criterion_id}
{scoring_context}
{rag_context}
Risposta dell'assessor:
{user_answer}

Analizza la risposta e restituisci un JSON con:
{{
  "score": <intero 0-3>,
  "confidence": "<H|M|L>",
  "evidence": "<sintesi dell'evidenza in max 200 caratteri>",
  "reasoning": "<breve spiegazione del punteggio assegnato>"
}}

Regole per la confidence:
  H = l'assessor cita documenti, metriche, risultati verificabili
  M = l'assessor descrive la situazione ma senza prove documentali
  L = l'assessor non ha dati o stima senza basi concrete

Sii conservativo: in caso di dubbio abbassa lo score o abbassa la confidence.
"""
        data = self._structured(prompt, _validate_criterion)

        return CriterionResponse(
            criterion_id=criterion_id,
            score=data["score"],
            confidence=data["confidence"],
            evidence=data["evidence"] or user_answer[:200],
            notes=data["reasoning"],
        )

    # ------------------------------------------------------------------
    # Fase 3: Finalizzazione e report narrativo
    # ------------------------------------------------------------------

    def finalize(
        self,
        session: AssessmentSession,
        assessed_dimensions: Optional[list[str]] = None,
    ) -> AssessmentSession:
        """Calcolo deterministico via engine; registra i metadati LLM."""
        session.llm_provider = self.provider.provider_name
        session.llm_model    = self.provider.model_name
        session.aipaf_version = _aipaf_version()
        return self.engine.compute(session, assessed_dimensions=assessed_dimensions)

    def generate_executive_summary(self, session: AssessmentSession) -> str:
        """Executive summary narrativo (con storico)."""
        if session.composite_score is None:
            return "Assessment non completato."

        def dim_line(d) -> str:
            if not d.assessed:
                return f"  {d.dimension_id} ({d.dimension_name}): NON VALUTATA in questa sessione"
            if not d.applicable:
                return f"  {d.dimension_id} ({d.dimension_name}): N/A (tutti i criteri non applicabili)"
            na = f"  [N/A: {', '.join(d.not_applicable)}]" if d.not_applicable else ""
            return f"  {d.dimension_id} ({d.dimension_name}): {d.raw_score:.2f}/3.00{na}"

        score_details = "\n".join(dim_line(d) for d in session.dimension_scores)

        active_patterns = [p for p in session.pattern_alerts if p.triggered]
        patterns_text = "\n".join(
            f"  - {p.label}: {p.description}" for p in active_patterns
        ) if active_patterns else "  Nessun pattern critico rilevato."

        flags_text = "\n".join(
            f"  - {f.norm_reference}: {f.description}"
            for f in session.normative_red_flags
        ) if session.normative_red_flags else "  Nessun flag normativo."

        gate = session.gate.value if session.gate else "N/A"
        partial_note = ""
        if session.partial:
            partial_note = (
                f"\nNOTA: assessment PARZIALE. Dimensioni non valutate: "
                f"{', '.join(session.unassessed_dimensions)}. Il gate provvisorio sulle "
                f"dimensioni valutate è {session.provisional_gate.value if session.provisional_gate else 'N/A'}; "
                f"non trarre conclusioni definitive di go/no-go.\n"
            )

        rag_context = ""
        if self.retriever:
            critical_dims = [d for d in session.dimension_scores
                             if d.counts and d.raw_score < REMEDIATION_THRESHOLD]
            rag_parts = []
            for dim in critical_dims[:3]:
                context = self.retriever.query(
                    f"{dim.dimension_id} {dim.dimension_name} requisiti obblighi",
                    n_results=2,
                )
                for chunk in context:
                    rag_parts.append(
                        f"[{chunk.source_type.upper()} · {chunk.filename}] {chunk.text[:300]}"
                    )
            if rag_parts:
                rag_context = "\nRiferimenti normativi pertinenti alle aree critiche:\n" + "\n\n".join(rag_parts) + "\n"

        prompt = f"""
Sei un consulente senior di AI Governance. Scrivi un executive summary professionale
in italiano per il seguente assessment AIPAF.

PROGETTO: {session.intake.project_name if session.intake else 'N/A'}
USE CASE: {session.intake.use_case_description if session.intake else ''}
SETTORE: {session.intake.sector.value if session.intake else ''}
EU AI ACT LEVEL: {session.intake.eu_ai_act_level.value if session.intake else ''}

RISULTATI:
  Composite Score: {session.composite_score:.2f}/3.00
  Gate: {gate}
{partial_note}
Score per dimensione:
{score_details}

Pattern di rischio rilevati:
{patterns_text}

Flag normativi:
{flags_text}
{rag_context}
Scrivi un executive summary di 3-4 paragrafi che:
1. Descriva sinteticamente il profilo del progetto e il gate ottenuto
2. Evidenzi i punti di forza principali (dimensioni con score >= 2.0)
3. Identifichi le aree critiche e i rischi principali
4. Indichi le priorita' di intervento piu' urgenti

Tono: professionale, diretto, orientato alle decisioni. Evita il gergo eccessivo.
"""
        return self._chat(prompt)

    def generate_remediation_plan(self, session: AssessmentSession) -> str:
        """Piano di remediation per le dimensioni valutate con score < 2.0."""
        weak_dims = [
            d for d in session.dimension_scores
            if d.counts and d.raw_score < REMEDIATION_THRESHOLD
        ]
        if not weak_dims:
            return "Nessun piano di remediation necessario: tutte le dimensioni valutate >= 2.0."

        weak_details = []
        for d in weak_dims:
            criteria_detail = "\n".join(
                f"    [{r.criterion_id}] score={r.effective_score:.1f} — {r.evidence[:100]}"
                for r in d.criteria_responses
                if not r.not_applicable and r.effective_score < REMEDIATION_THRESHOLD
            )
            weak_details.append(
                f"  {d.dimension_id} — {d.dimension_name} (score: {d.raw_score:.2f}):\n{criteria_detail}"
            )

        prompt = f"""
Progetto: {session.intake.project_name if session.intake else 'N/A'}
Gate: {session.gate.value if session.gate else 'N/A'}

Dimensioni con score insufficiente:
{chr(10).join(weak_details)}

Genera un piano di remediation strutturato con:
- Per ogni dimensione critica: 2-3 azioni concrete con priorita' (Alta/Media/Bassa)
- Stima dell'effort (giorni/settimane)
- Dipendenze tra azioni
- Criterio di completamento verificabile

Formato: lista strutturata per dimensione, professionale e actionable.
"""
        return self._chat(prompt)


# ---------------------------------------------------------------------------
# Validatori output strutturato
# ---------------------------------------------------------------------------

def _validate_intake(data: dict) -> dict:
    def text(key: str, default: str) -> str:
        v = data.get(key)
        return str(v).strip() if v not in (None, "") else default

    try:
        sector = Sector(str(data.get("sector", "")).strip().lower())
    except ValueError:
        raise LLMOutputError(
            f"sector non ammesso: {data.get('sector')!r} (ammessi: {[s.value for s in Sector]})"
        )
    try:
        level = EUAIActLevel(str(data.get("eu_ai_act_level", "")).strip().upper())
    except ValueError:
        raise LLMOutputError(
            f"eu_ai_act_level non ammesso: {data.get('eu_ai_act_level')!r} "
            f"(ammessi: {[l.value for l in EUAIActLevel]})"
        )
    name = text("project_name", "")
    if not name:
        raise LLMOutputError("project_name mancante nell'intake")

    users = data.get("user_count_estimate")
    try:
        users = int(users) if users not in (None, "", "null") else None
    except (TypeError, ValueError):
        users = None

    return {
        "project_name":           name,
        "use_case_description":   text("use_case_description", ""),
        "project_owner":          text("project_owner", "Da definire"),
        "sector":                 sector,
        "ai_system_type":         text("ai_system_type", ""),
        "eu_ai_act_level":        level,
        "project_stage":          text("project_stage", ""),
        "user_count_estimate":    users,
        "vulnerable_subjects":    _to_bool(data.get("vulnerable_subjects")),
        "personal_data_treated":  _to_bool(data.get("personal_data_treated")),
        "sensitive_data_treated": _to_bool(data.get("sensitive_data_treated")),
    }


def _validate_criterion(data: dict) -> dict:
    raw_score = data.get("score")
    try:
        score = int(raw_score)
    except (TypeError, ValueError):
        raise LLMOutputError(f"score non intero: {raw_score!r}")
    if not 0 <= score <= 3:
        raise LLMOutputError(f"score fuori scala 0-3: {score}")
    try:
        confidence = ConfidenceLevel(str(data.get("confidence", "")).strip().upper())
    except ValueError:
        raise LLMOutputError(f"confidence non ammessa: {data.get('confidence')!r} (ammesse: H, M, L)")
    return {
        "score":      score,
        "confidence": confidence,
        "evidence":   str(data.get("evidence") or "").strip(),
        "reasoning":  str(data.get("reasoning") or "").strip(),
    }


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "si", "sì", "1")
    return bool(v)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> dict:
    """
    Estrae e parsa il primo oggetto JSON da una risposta LLM.
    Solleva LLMOutputError se non c'è un oggetto JSON valido: nessun
    fallback silenzioso, perché un dict vuoto diventerebbe uno score reale.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "")
    cleaned = re.sub(r"```", "", cleaned).strip()

    start = cleaned.find("{")
    end   = cleaned.rfind("}") + 1
    if start < 0 or end <= start:
        raise LLMOutputError(f"Nessun oggetto JSON nella risposta: {cleaned[:200]!r}")

    try:
        data = json.loads(cleaned[start:end])
    except json.JSONDecodeError as e:
        raise LLMOutputError(f"JSON malformato ({e}): {cleaned[start:start+200]!r}")
    if not isinstance(data, dict):
        raise LLMOutputError(f"Atteso oggetto JSON, ricevuto {type(data).__name__}")
    return data


def _today() -> str:
    from datetime import date
    return date.today().isoformat()


def _aipaf_version() -> str:
    try:
        from . import __version__
        return __version__
    except Exception:
        return "unknown"
