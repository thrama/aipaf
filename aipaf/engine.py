"""
AIPAF — AI Project Assessment Framework
engine.py — Scoring Engine puro Python (zero dipendenze AI)

Regole di aggregazione (v0.2.0):
  - Criteri N/A: esclusi dal denominatore della dimensione.
  - Dimensione con tutti i criteri N/A: applicable=False, esclusa da composito e gate.
  - Dimensioni non valutate (assessment parziale): assessed=False, escluse da
    composito, gate e pattern alert. Il gate risultante è PARTIAL, con
    provisional_gate calcolato sulle sole dimensioni valutate.
  - Le soglie vivono in models.py (single source of truth).

Autore: Lorenzo Lombardi
"""

from __future__ import annotations
import json
import logging
import os
from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

from .models import (
    DIMENSION_CRITICAL_THRESHOLD, GATE_GREEN_THRESHOLD,
    GATE_YELLOW_THRESHOLD, ApprovalGate, AssessmentSession, CriterionResponse,
    DimensionScore, EUAIActLevel, FrameworkConfig, NormativeRedFlag,
    PatternAlert, ProjectIntake, Sector, SectorWeights,
)

log = logging.getLogger("aipaf.engine")

_CONFIG_RESOURCE = "AIPAF_framework_config.json"


# ---------------------------------------------------------------------------
# Caricamento configurazione
# ---------------------------------------------------------------------------

def default_config_path() -> Path:
    """
    Percorso del config JSON, in ordine di priorità:
      1. variabile d'ambiente AIPAF_CONFIG_PATH
      2. risorsa impacchettata aipaf/config/AIPAF_framework_config.json
    """
    env = os.environ.get("AIPAF_CONFIG_PATH")
    if env:
        return Path(env)
    return Path(str(resources.files("aipaf").joinpath("config", _CONFIG_RESOURCE)))


def load_config(path: Optional[Path] = None) -> FrameworkConfig:
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Config AIPAF non trovato: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = json.load(f)
    return FrameworkConfig(
        dimensions=raw["dimensions"],
        sector_weights=raw["sector_weights"]["sectors"],
        pattern_alerts=raw["pattern_alerts"],
        framework_version=str(raw.get("framework", {}).get("version", "")),
    )


# ---------------------------------------------------------------------------
# skip_condition
# ---------------------------------------------------------------------------

def _parse_literal(raw: str):
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
        return raw[1:-1]
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def evaluate_skip_condition(condition: Optional[str], intake: Optional[ProjectIntake]) -> bool:
    """
    Valuta una skip_condition ('<campo> == <valore>' | '<campo> != <valore>')
    contro i campi dell'intake. True → il criterio è candidato N/A.

    Solleva ValueError se la sintassi o il campo non sono validi: una condizione
    scritta male nel config non deve fallire silenziosamente.
    """
    if not condition or intake is None:
        return False
    for op in ("!=", "=="):
        if op in condition:
            field, raw_value = condition.split(op, 1)
            field = field.strip()
            if field not in ProjectIntake.model_fields:
                raise ValueError(
                    f"skip_condition non valida {condition!r}: campo intake sconosciuto {field!r}"
                )
            actual   = getattr(intake, field)
            expected = _parse_literal(raw_value)
            if hasattr(actual, "value"):     # Enum → confronto sul valore
                actual = actual.value
            return (actual == expected) if op == "==" else (actual != expected)
    raise ValueError(f"skip_condition non valida {condition!r}: operatore atteso '==' o '!='")


# ---------------------------------------------------------------------------
# ScoringEngine
# ---------------------------------------------------------------------------

class ScoringEngine:

    def __init__(self, config: FrameworkConfig):
        self.config = config

    # ------------------------------------------------------------------
    # API pubblica
    # ------------------------------------------------------------------

    def compute(
        self,
        session: AssessmentSession,
        assessed_dimensions: Optional[Iterable[str]] = None,
    ) -> AssessmentSession:
        """
        Calcola dimension score, composito, gate, pattern alert e red flag normativi.

        Args:
            assessed_dimensions: dimensioni effettivamente valutate in questa sessione.
                Se None, usa session.assessed_dimensions; se anche questa è vuota,
                le deduce dalle risposte (una dimensione è valutata se ha almeno
                una risposta).
        """
        self._validate_responses(session)

        if assessed_dimensions is not None:
            assessed = {d.upper() for d in assessed_dimensions}
        elif session.assessed_dimensions:
            assessed = set(session.assessed_dimensions)
        else:
            assessed = {r.criterion_id.split(".")[0] for r in session.responses}
        assessed &= set(self.config.dimension_ids)

        session.assessed_dimensions = [d for d in self.config.dimension_ids if d in assessed]
        session.partial = set(session.assessed_dimensions) != set(self.config.dimension_ids)

        sector  = session.intake.sector if session.intake else Sector.DEFAULT
        weights = self.config.sector_weights.get(sector.value) or self.config.sector_weights["default"]

        session.dimension_scores = self._compute_dimension_scores(session, weights, assessed)
        session.composite_score  = self._compute_composite(session.dimension_scores)

        gate = self._compute_gate(session.dimension_scores, session.composite_score)
        if session.partial:
            session.gate             = ApprovalGate.PARTIAL
            session.provisional_gate = gate
        else:
            session.gate             = gate
            session.provisional_gate = None

        session.pattern_alerts = self._evaluate_pattern_alerts(session)
        eu_level = session.intake.eu_ai_act_level if session.intake else EUAIActLevel.MINIMAL_RISK
        session.normative_red_flags = self._evaluate_normative_flags(session, eu_level)
        return session

    def candidate_not_applicable(self, intake: Optional[ProjectIntake]) -> list[str]:
        """ID dei criteri la cui skip_condition è vera per questo intake."""
        out: list[str] = []
        for dim in self.config.dimensions:
            for c in dim.criteria:
                if evaluate_skip_condition(c.skip_condition, intake):
                    out.append(c.id)
        return out

    # ------------------------------------------------------------------
    # Validazione
    # ------------------------------------------------------------------

    def _validate_responses(self, session: AssessmentSession) -> None:
        known = self.config.criterion_ids
        unknown = [r.criterion_id for r in session.responses if r.criterion_id not in known]
        if unknown:
            raise ValueError(
                f"criterion_id non presenti nel framework: {', '.join(unknown)}"
            )

    # ------------------------------------------------------------------
    # Dimensioni
    # ------------------------------------------------------------------

    def _compute_dimension_scores(
        self,
        session:  AssessmentSession,
        weights:  SectorWeights,
        assessed: set[str],
    ) -> list[DimensionScore]:
        weight_map = weights.as_map()
        scores: list[DimensionScore] = []
        for dim_cfg in self.config.dimensions:
            dim_responses = [r for r in session.responses
                             if r.criterion_id.startswith(dim_cfg.id + ".")]
            if dim_cfg.id not in assessed:
                scores.append(DimensionScore(
                    dimension_id=dim_cfg.id, dimension_name=dim_cfg.name,
                    raw_score=0.0, weight_pct=weight_map[dim_cfg.id],
                    assessed=False, criteria_responses=dim_responses,
                ))
                continue

            raw_score, applicable, flags, data_needed, na = \
                self._aggregate_dimension(dim_cfg.id, dim_responses)
            scores.append(DimensionScore(
                dimension_id=dim_cfg.id, dimension_name=dim_cfg.name,
                raw_score=raw_score, weight_pct=weight_map[dim_cfg.id],
                assessed=True, applicable=applicable,
                criteria_responses=dim_responses, flags=flags,
                data_needed=data_needed, not_applicable=na,
            ))
        return scores

    def _aggregate_dimension(
        self, dim_id: str, responses: list[CriterionResponse],
    ) -> tuple[float, bool, list[str], list[str], list[str]]:
        """
        Media pesata sui soli criteri applicabili.
        Returns: (raw_score, applicable, flags, data_needed, not_applicable_ids)
        """
        dim_cfg = self.config.dimension_map[dim_id]
        weight_map = {c.id: c.weight_pct for c in dim_cfg.criteria}

        total_weight = 0
        weighted_sum = 0.0
        flags: list[str] = []
        data_needed: list[str] = []
        na: list[str] = []

        for resp in responses:
            if resp.not_applicable:
                na.append(resp.criterion_id)
                continue
            w = weight_map[resp.criterion_id]
            weighted_sum += resp.effective_score * w
            total_weight += w
            crit_cfg = self.config.get_criterion(resp.criterion_id)
            if crit_cfg and resp.score <= 1:
                for flag in crit_cfg.flags:
                    flags.append(f"[{resp.criterion_id}] {flag}")
            if resp.data_needed_flag:
                data_needed.append(resp.criterion_id)

        if total_weight == 0:
            # Nessun criterio applicabile con peso: se ci sono risposte sono tutte N/A
            applicable = not (responses and len(na) == len(responses))
            return 0.0, applicable, flags, data_needed, na

        raw_score = min(weighted_sum / total_weight, 3.0)
        return round(raw_score, 3), True, flags, data_needed, na

    # ------------------------------------------------------------------
    # Composito e gate
    # ------------------------------------------------------------------

    def _compute_composite(self, dimension_scores: list[DimensionScore]) -> float:
        counting = [d for d in dimension_scores if d.counts]
        total_weight = sum(d.weight_pct for d in counting)
        if total_weight == 0:
            return 0.0
        return round(sum(d.raw_score * d.weight_pct for d in counting) / total_weight, 3)

    def _compute_gate(self, dimension_scores: list[DimensionScore], composite_score: float) -> ApprovalGate:
        counting = [d for d in dimension_scores if d.counts]
        if not counting:
            return ApprovalGate.RED
        for d in counting:
            if d.raw_score < DIMENSION_CRITICAL_THRESHOLD:
                return ApprovalGate.RED
        if composite_score < GATE_YELLOW_THRESHOLD:
            return ApprovalGate.RED
        if composite_score < GATE_GREEN_THRESHOLD:
            return ApprovalGate.YELLOW
        return ApprovalGate.GREEN

    # ------------------------------------------------------------------
    # Pattern alert
    # ------------------------------------------------------------------

    def _evaluate_pattern_alerts(self, session: AssessmentSession) -> list[PatternAlert]:
        patterns = self.config.pattern_alerts.get("patterns", [])
        # Solo dimensioni che concorrono: una dimensione non valutata o N/A
        # non deve far scattare pattern per assenza di dati.
        dim_score_map  = {d.dimension_id: d.raw_score for d in session.dimension_scores if d.counts}
        assessed       = set(session.assessed_dimensions)
        crit_score_map = {
            r.criterion_id: r.effective_score for r in session.responses
            if not r.not_applicable and r.criterion_id.split(".")[0] in assessed
        }
        alerts: list[PatternAlert] = []
        for pattern in patterns:
            triggered = self._eval_condition(pattern["condition"], dim_score_map, crit_score_map)
            alerts.append(PatternAlert(
                pattern_id=pattern["id"], name=pattern["name"], label=pattern["label"],
                description=pattern["description"], recommended_action=pattern["recommended_action"],
                triggered=triggered,
            ))
        return alerts

    def _eval_condition(self, condition: str, dim_scores: dict, crit_scores: dict) -> bool:
        return all(self._eval_single(p.strip(), dim_scores, crit_scores) for p in condition.split("AND"))

    def _eval_single(self, expr: str, dim_scores: dict, crit_scores: dict) -> bool:
        for op in (">=", "<=", ">", "<", "=="):
            if op in expr:
                left, right = expr.split(op, 1)
                left = left.strip()
                value = float(right.strip())
                score = crit_scores.get(left) if "." in left else dim_scores.get(left)
                if score is None:
                    return False
                return {">=": score >= value, "<=": score <= value, ">": score > value,
                        "<": score < value, "==": score == value}[op]
        raise ValueError(f"Condizione pattern non valida: {expr!r}")

    # ------------------------------------------------------------------
    # Red flag normativi
    # ------------------------------------------------------------------

    def _evaluate_normative_flags(self, session: AssessmentSession, eu_level: EUAIActLevel) -> list[NormativeRedFlag]:
        flags: list[NormativeRedFlag] = []
        assessed = set(session.assessed_dimensions)

        def s(cid: str) -> Optional[float]:
            """None se la dimensione non è valutata o il criterio è N/A: nessun flag."""
            if cid.split(".")[0] not in assessed:
                return None
            r = session.get_response(cid)
            if r is not None and r.not_applicable:
                return None
            return r.effective_score if r else 0.0

        def below(cid: str, threshold: float) -> bool:
            v = s(cid)
            return v is not None and v < threshold

        if eu_level == EUAIActLevel.HIGH_RISK:
            if below("D5.C3", 2):
                flags.append(NormativeRedFlag(
                    criterion_id="D5.C3", criterion_name="Human-in-the-loop",
                    norm_reference="Art. 14 EU AI Act",
                    description="Sistema HIGH RISK: supervisione umana obbligatoria (Art. 14). Score D5.C3 insufficiente.",
                ))
            if below("D4.C4", 2):
                flags.append(NormativeRedFlag(
                    criterion_id="D4.C4", criterion_name="Spiegabilita' e riproducibilita'",
                    norm_reference="Art. 13 EU AI Act",
                    description="Sistema HIGH RISK: trasparenza obbligatoria (Art. 13). Score D4.C4 insufficiente.",
                ))
        if eu_level in (EUAIActLevel.LIMITED_RISK, EUAIActLevel.HIGH_RISK):
            if below("D5.C2", 2):
                flags.append(NormativeRedFlag(
                    criterion_id="D5.C2", criterion_name="Trasparenza e comunicazione",
                    norm_reference="Art. 50 EU AI Act",
                    description="Sistema LIMITED/HIGH RISK: obbligo informare utenti (Art. 50). Score D5.C2 insufficiente.",
                ))
        return flags

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def dimension_score_for(self, session: AssessmentSession, dim_id: str) -> Optional[float]:
        for d in session.dimension_scores:
            if d.dimension_id == dim_id:
                return d.raw_score
        return None

    def summary(self, session: AssessmentSession) -> str:
        if session.composite_score is None:
            return "Assessment non ancora calcolato."
        gate = session.gate.value if session.gate else "N/A"
        gate_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴", "PARTIAL": "⚪"}
        gate_line = f"  Gate            : {gate_emoji.get(gate, '')} {gate}"
        if session.partial and session.provisional_gate:
            gate_line += f"  (provvisorio sulle dimensioni valutate: {session.provisional_gate.value})"
        lines = [
            "=" * 56,
            f"  AIPAF  |  {session.intake.project_name if session.intake else 'N/A'}",
            "=" * 56,
            f"  Composite Score : {session.composite_score:.2f} / 3.00",
            gate_line,
        ]
        if session.partial:
            lines.append(f"  Copertura       : PARZIALE — non valutate: {', '.join(session.unassessed_dimensions)}")
        if session.llm_provider:
            lines.append(f"  LLM             : {session.llm_provider}/{session.llm_model or '?'}")
        lines += ["─" * 56, "  Dimensioni:"]
        for d in session.dimension_scores:
            if not d.assessed:
                lines.append(f"    {d.dimension_id}  {'[ non valutata ]':<12}          (peso {d.weight_pct}%)")
                continue
            if not d.applicable:
                lines.append(f"    {d.dimension_id}  {'[ N/A ]':<12}          (peso {d.weight_pct}%)  tutti i criteri non applicabili")
                continue
            critical = "  ⚠ CRITICO" if d.is_critical else ""
            na = f"  N/A: {', '.join(d.not_applicable)}" if d.not_applicable else ""
            lines.append(f"    {d.dimension_id}  {self._bar(d.raw_score)}  {d.raw_score:.2f}  (peso {d.weight_pct}%){critical}{na}")
        active = [p for p in session.pattern_alerts if p.triggered]
        if active:
            lines += ["─" * 56, "  Pattern Alert:"]
            for p in active:
                lines.append(f"    ⚡ {p.label} — {p.name}")
        if session.normative_red_flags:
            lines += ["─" * 56, "  Red Flag Normativi:"]
            for f in session.normative_red_flags:
                lines.append(f"    🚨 {f.norm_reference} — {f.criterion_id}")
        lines.append("=" * 56)
        return "\n".join(lines)

    @staticmethod
    def _bar(score: float, width: int = 10) -> str:
        filled = int(round(score / 3.0 * width))
        return "[" + "█" * filled + "░" * (width - filled) + "]"
