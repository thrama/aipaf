"""
AIPAF — AI Project Assessment Framework
cli.py — Interfaccia a riga di comando (Typer)

Comandi disponibili:
  aipaf assess   — avvia un assessment interattivo
  aipaf score    — calcola score da un file JSON di risposte (batch)
  aipaf report   — genera report Markdown da una sessione salvata
  aipaf summary  — stampa il summary di una sessione JSON salvata
  aipaf check    — verifica configurazione e provider LLM
  aipaf rag      — gestione RAG layer (ingest, status)

Uso rapido:
  aipaf assess --llm ollama --model qwen3.5:4b
  aipaf assess --llm claude --dims D2,D3
  aipaf score risposte.json --no-narrative
  aipaf check

Autore: Lorenzo Lombardi
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import click
import typer
from typing_extensions import Annotated
from dotenv import load_dotenv

from aipaf import __version__
from aipaf.agent import AssessmentAgent, LLMOutputError
from aipaf.engine import ScoringEngine, default_config_path, evaluate_skip_condition, load_config
from aipaf.llm.factory import get_provider, available_providers
from aipaf.models import (
    REMEDIATION_THRESHOLD,
    AssessmentSession, ConfidenceLevel, CriterionResponse,
)

app = typer.Typer(
    name="aipaf",
    help=f"AIPAF — AI Project Assessment Framework CLI (v{__version__})",
    no_args_is_help=True,
)

rag_app = typer.Typer(name="rag", help="Gestione RAG layer (ingestione documenti)")
app.add_typer(rag_app)

logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(levelname)s: %(message)s")

# .env dalla cartella corrente (o superiori): ANTHROPIC_API_KEY, AIPAF_CLAUDE_MODEL, OLLAMA_HOST, ...
# Le variabili gia' presenti nell'ambiente hanno la precedenza (override=False).
load_dotenv(override=False)

_SKIP_WORDS = ("skip", "s")
_NA_WORDS   = ("na", "n/a", "n.a.")

# ---------------------------------------------------------------------------
# Helpers UI
# ---------------------------------------------------------------------------

def _print_header():
    typer.echo(typer.style(
        "\n╔══════════════════════════════════════════════════╗\n"
        f"║  AIPAF — AI Project Assessment Framework  v{__version__:<6}║\n"
        "║  Lorenzo Lombardi                               ║\n"
        "╚══════════════════════════════════════════════════╝",
        fg=typer.colors.CYAN, bold=True
    ))

def _ok(msg: str):
    typer.echo(typer.style(f"✓ {msg}", fg=typer.colors.GREEN))

def _warn(msg: str):
    typer.echo(typer.style(f"⚠  {msg}", fg=typer.colors.YELLOW))

def _err(msg: str):
    typer.echo(typer.style(f"✗ {msg}", fg=typer.colors.RED), err=True)

def _section(title: str):
    typer.echo(f"\n{'─'*50}")
    typer.echo(typer.style(f"  {title}", bold=True))
    typer.echo('─'*50)

def _ask(prompt: str, default: Optional[str] = None) -> str:
    return typer.prompt(typer.style(prompt, fg=typer.colors.BRIGHT_WHITE), default=default)

def _describe_provider(prov) -> None:
    """Esito health check con messaggi specifici per provider."""
    typer.echo(f"  Provider: {prov.provider_name} / {prov.model_name}")
    if prov.provider_name == "ollama":
        state = prov.model_available()
        if state is None:
            _warn("Ollama non raggiungibile (avviare con: ollama serve)")
        elif state is False:
            _warn(f"Modello '{prov.model_name}' non presente in Ollama. "
                  f"Eseguire: ollama pull {prov.model_name}")
        else:
            _ok("Ollama raggiungibile, modello disponibile")
    else:
        if prov.health_check():
            _ok(f"Provider '{prov.provider_name}' raggiungibile")
        else:
            _warn(f"Provider '{prov.provider_name}' non raggiungibile")
            if getattr(prov, "last_error", None):
                typer.echo(f"    Motivo: {prov.last_error}")

def _load_session_file(path: Path) -> AssessmentSession:
    if not path.exists():
        _err(f"File non trovato: {path}")
        raise typer.Exit(1)
    try:
        with open(path, encoding="utf-8") as f:
            return AssessmentSession.model_validate(json.load(f))
    except Exception as e:
        _err(f"Errore lettura sessione {path}: {e}")
        raise typer.Exit(1)

def _save_session(session: AssessmentSession, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(session.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

def _partial_warning(session: AssessmentSession) -> None:
    if session.partial:
        _warn(f"Assessment PARZIALE: dimensioni non valutate {', '.join(session.unassessed_dimensions)}. "
              "Il gate non e' definitivo.")


# ---------------------------------------------------------------------------
# Comando: check
# ---------------------------------------------------------------------------

@app.command()
def check(
    llm: Annotated[str, typer.Option(help="Provider LLM da verificare")] = "ollama",
    model: Annotated[Optional[str], typer.Option(help="Modello da usare")] = None,
    config_path: Annotated[Optional[Path], typer.Option(help="Percorso config JSON")] = None,
):
    """Verifica configurazione framework e connettivita' provider LLM."""
    _print_header()
    typer.echo("\nVerifica configurazione in corso...\n")

    try:
        cfg = load_config(config_path)
        n_dims = len(cfg.dimensions)
        n_crit = sum(len(d.criteria) for d in cfg.dimensions)
        _ok(f"Config caricato ({config_path or default_config_path()}) — "
            f"framework v{cfg.framework_version or '?'}, {n_dims} dimensioni, {n_crit} criteri")
    except Exception as e:
        _err(f"Errore caricamento config: {e}")
        raise typer.Exit(1)

    try:
        prov = get_provider(llm, model=model)
        _describe_provider(prov)
    except Exception as e:
        _err(f"Errore provider {llm!r}: {e}")

    typer.echo(f"\n  Provider disponibili: {', '.join(available_providers())}")


# ---------------------------------------------------------------------------
# Comando: assess (interattivo)
# ---------------------------------------------------------------------------

@app.command()
def assess(
    llm: Annotated[str, typer.Option(help="Provider LLM: claude | ollama")] = "ollama",
    model: Annotated[Optional[str], typer.Option(help="Modello LLM")] = None,
    config_path: Annotated[Optional[Path], typer.Option(help="Percorso config JSON")] = None,
    output: Annotated[Optional[Path], typer.Option(help="Salva sessione JSON in questo file")] = None,
    verbose: Annotated[bool, typer.Option(help="Mostra dettagli chiamate LLM")] = False,
    dims: Annotated[Optional[str], typer.Option(
        help="Dimensioni da valutare (es. D1,D2,D3). Default: tutte. Con un sottoinsieme il gate e' PARTIAL"
    )] = None,
    rag: Annotated[bool, typer.Option(help="Abilita RAG layer (richiede documenti indicizzati)")] = False,
    rag_backend: Annotated[Optional[str], typer.Option(
        help="Backend embedding RAG: ollama | hash. Default: quello con cui l'indice e' stato creato"
    )] = None,
    report: Annotated[Optional[Path], typer.Option(help="Percorso output report .md")] = None,
):
    """
    Avvia un assessment interattivo guidato dall'LLM.

    Per ogni criterio: descrivi la situazione, oppure 'skip' (registra confidence L,
    score 0) oppure 'na' (non applicabile, escluso dal calcolo).
    """
    _print_header()

    # Setup
    try:
        cfg    = load_config(config_path)
        engine = ScoringEngine(cfg)
        prov   = get_provider(llm, model=model)

        retriever = None
        if rag:
            try:
                from aipaf.rag.ingestor import DocumentIngestor
                from aipaf.rag.retriever import RAGRetriever
                ingestor  = DocumentIngestor(embedding_backend=rag_backend)
                retriever = RAGRetriever(ingestor)
                rag_status = ingestor.status()
                n_reg  = rag_status["regulatory"]["chunks"]
                n_comp = rag_status["company"]["chunks"]
                _ok(f"RAG attivo — {n_reg} chunk normativi, {n_comp} chunk aziendali")
                if n_reg + n_comp == 0:
                    _warn("Nessun documento indicizzato. Esegui: aipaf rag ingest")
            except Exception as e:
                _warn(f"RAG non disponibile: {e}")

        agent = AssessmentAgent(
            provider=prov, config=cfg, engine=engine,
            retriever=retriever, verbose=verbose,
        )
    except Exception as e:
        _err(f"Inizializzazione fallita: {e}")
        raise typer.Exit(1)

    typer.echo(f"\n  Provider: {typer.style(prov.provider_name+'/'+prov.model_name, bold=True)}")

    # Dimensioni da valutare
    selected_dims: Optional[list[str]] = None
    if dims:
        requested = [d.strip().upper() for d in dims.split(",") if d.strip()]
        unknown = [d for d in requested if d not in cfg.dimension_ids]
        if unknown:
            _err(f"Dimensioni sconosciute: {', '.join(unknown)} (valide: {', '.join(cfg.dimension_ids)})")
            raise typer.Exit(1)
        selected_dims = [d for d in cfg.dimension_ids if d in requested]
        if set(selected_dims) != set(cfg.dimension_ids):
            _warn(f"Assessment parziale su {', '.join(selected_dims)}: il gate risultante sara' PARTIAL.")

    session = AssessmentSession()

    # ── Fase 1: Intake ──────────────────────────────────────────────────────
    _section("FASE 1 — Intake progetto")
    typer.echo(
        "Descrivi il progetto AI da valutare: obiettivo, settore, stadio attuale,\n"
        "tipo di dati trattati, utenti coinvolti.\n"
    )
    placeholder = "Inserisci qui la descrizione del progetto..."
    try:
        description = click.edit(placeholder)
    except Exception:
        description = None   # nessun editor disponibile (es. Windows senza EDITOR)
    if not description or description.strip() == placeholder:
        description = _ask("Descrizione progetto")

    typer.echo("\nAnalisi del progetto in corso...")
    try:
        session = agent.collect_intake(session, description)
    except LLMOutputError as e:
        _err(f"Intake non interpretabile dall'LLM: {e}")
        _err("Senza settore e livello EU AI Act validi l'assessment non puo' procedere. "
             "Riprovare con una descrizione piu' esplicita o con un modello piu' capace.")
        raise typer.Exit(2)
    except Exception as e:
        _err(f"Errore provider durante l'intake: {e}")
        raise typer.Exit(2)
    intake = session.intake

    typer.echo(f"\n  Progetto  : {typer.style(intake.project_name, bold=True)}")
    typer.echo(f"  Settore   : {intake.sector.value}")
    typer.echo(f"  Stadio    : {intake.project_stage}")
    typer.echo(f"  EU AI Act : {intake.eu_ai_act_level.value}")
    typer.echo(f"  Dati pers.: {'Sì' if intake.personal_data_treated else 'No'}")
    typer.echo(f"  Dati sens.: {'Sì' if intake.sensitive_data_treated else 'No'}")

    if not typer.confirm("\nL'intake e' corretto? Procedere con l'assessment?", default=True):
        _warn("Assessment interrotto.")
        raise typer.Exit(0)

    # ── Fase 2: Assessment dimensioni ───────────────────────────────────────
    _section("FASE 2 — Assessment multidimensionale")
    typer.echo("Risposta a ogni criterio in linguaggio naturale.\n"
               "Indicare le evidenze disponibili per aumentare la confidence.\n"
               "'skip' = non valutato (confidence L, score 0)   'na' = non applicabile (escluso)\n")

    failed_criteria: list[str] = []

    for dim_cfg in cfg.dimensions:
        if selected_dims is not None and dim_cfg.id not in selected_dims:
            continue

        _section(f"{dim_cfg.id} — {dim_cfg.name}")
        typer.echo(typer.style(f"  {dim_cfg.implicit_purpose}", fg=typer.colors.BRIGHT_BLACK))
        typer.echo(f"  Compilato da: {dim_cfg.compiled_by}\n")

        for crit in dim_cfg.criteria:
            typer.echo(f"\n{'·'*50}")
            typer.echo(typer.style(
                f"  [{crit.id}] {crit.name}  (peso {crit.weight_pct}%)",
                bold=True
            ))
            typer.echo(f"  {crit.description}")
            typer.echo(f"  Fonte: {typer.style(crit.source, fg=typer.colors.BRIGHT_BLACK)}")
            typer.echo(
                f"\n  Scala: "
                f"3={crit.scoring_guide.s3[:60]}... | "
                f"0={crit.scoring_guide.s0[:40]}..."
            )

            # N/A automatico da skip_condition (con conferma)
            try:
                auto_na = evaluate_skip_condition(crit.skip_condition, intake)
            except ValueError as e:
                _warn(f"skip_condition non valutabile per {crit.id}: {e}")
                auto_na = False
            if auto_na:
                _warn(f"  In base all'intake ({crit.skip_condition}) il criterio risulta non applicabile.")
                if typer.confirm(f"  Marcare {crit.id} come N/A?", default=True):
                    session.upsert_response(CriterionResponse(
                        criterion_id=crit.id, not_applicable=True,
                        evidence=f"Non applicabile: {crit.skip_condition}",
                    ))
                    typer.echo(typer.style("  → N/A (escluso dal calcolo)", fg=typer.colors.BRIGHT_BLACK))
                    continue

            answer = _ask(f"\n  {crit.id} › Descrivi la situazione ('skip' | 'na')")
            token = answer.strip().lower()

            if token in _NA_WORDS:
                reason = _ask("  Motivazione N/A", default="Non applicabile al progetto")
                session.upsert_response(CriterionResponse(
                    criterion_id=crit.id, not_applicable=True, evidence=reason,
                ))
                typer.echo(typer.style("  → N/A (escluso dal calcolo)", fg=typer.colors.BRIGHT_BLACK))
                continue

            if token in _SKIP_WORDS or not token:
                _warn(f"  {crit.id} saltato — confidence L (score = 0, data needed)")
                session.upsert_response(CriterionResponse(
                    criterion_id=crit.id, score=0, confidence=ConfidenceLevel.LOW,
                    evidence="Criterio saltato durante l'assessment",
                ))
                continue

            # Interpretazione LLM — un fallimento non interrompe la sessione
            typer.echo(typer.style("  Interpretazione in corso...", fg=typer.colors.BRIGHT_BLACK))
            try:
                resp = agent.interpret_criterion_response(crit.id, answer, dim_cfg)
            except (LLMOutputError, ConnectionError, TimeoutError, RuntimeError) as e:
                _warn(f"  Interpretazione LLM fallita per {crit.id}: {e}")
                _warn("  Registrato in modo conservativo: confidence L, score 0. "
                      "Rivedere il criterio prima della review.")
                failed_criteria.append(crit.id)
                resp = CriterionResponse(
                    criterion_id=crit.id, score=0, confidence=ConfidenceLevel.LOW,
                    evidence=answer[:200],
                    notes=f"INTERPRETAZIONE LLM FALLITA: {e}",
                )
            session.upsert_response(resp)

            score_colors = {3: typer.colors.GREEN, 2: typer.colors.CYAN,
                            1: typer.colors.YELLOW, 0: typer.colors.RED}
            score_str = typer.style(
                f"Score: {resp.score}/3  Confidence: {resp.confidence.value}",
                fg=score_colors.get(resp.score, typer.colors.WHITE), bold=True,
            )
            typer.echo(f"\n  → {score_str}")
            if resp.notes:
                typer.echo(f"    {resp.notes[:120]}")

    # ── Fase 3: Scoring e report ─────────────────────────────────────────────
    _section("FASE 3 — Scoring e gate decision")
    typer.echo("Calcolo score composito...")

    session = agent.finalize(session, assessed_dimensions=selected_dims)
    typer.echo("\n" + engine.summary(session))
    _partial_warning(session)
    if failed_criteria:
        _warn(f"Criteri con interpretazione LLM fallita (confidence L): {', '.join(failed_criteria)}")

    exec_summary = ""
    if typer.confirm("\nGenerare executive summary narrativo con LLM?", default=True):
        typer.echo("\nGenerazione in corso...")
        try:
            exec_summary = agent.generate_executive_summary(session)
            _section("EXECUTIVE SUMMARY")
            typer.echo(exec_summary)
        except Exception as e:
            _warn(f"Executive summary non generato: {e}")

    remediation = ""
    has_weak = any(d.counts and d.raw_score < REMEDIATION_THRESHOLD for d in session.dimension_scores)
    if has_weak and typer.confirm("\nGenerare piano di remediation?", default=True):
        typer.echo("\nGenerazione in corso...")
        try:
            remediation = agent.generate_remediation_plan(session)
            _section("REMEDIATION PLAN")
            typer.echo(remediation)
        except Exception as e:
            _warn(f"Remediation plan non generato: {e}")

    from aipaf.report import ReportGenerator
    gen = ReportGenerator()
    if report:
        rep_path = gen.generate_and_save(session, exec_summary, remediation, output_path=report)
        _ok(f"Report salvato in: {rep_path}")
    elif typer.confirm("\nGenerare report Markdown?", default=True):
        rep_path = gen.generate_and_save(session, exec_summary, remediation)
        _ok(f"Report salvato in: {rep_path}")

    if output:
        _save_session(session, output)
        _ok(f"Sessione salvata in: {output}")
    elif typer.confirm("\nSalvare la sessione JSON?", default=False):
        default_path = Path(f"assessment_{intake.project_name.lower().replace(' ','_')}.json")
        out_path = Path(_ask("Percorso file", default=str(default_path)))
        _save_session(session, out_path)
        _ok(f"Sessione salvata in: {out_path}")

    typer.echo("\n")
    _ok("Assessment completato." if not session.partial else "Assessment parziale completato.")


# ---------------------------------------------------------------------------
# Comando: score (batch da JSON)
# ---------------------------------------------------------------------------

@app.command()
def score(
    input: Annotated[Path, typer.Argument(help="File JSON con le risposte pre-compilate")],
    llm: Annotated[str, typer.Option(help="Provider LLM per il report narrativo")] = "ollama",
    model: Annotated[Optional[str], typer.Option(help="Modello LLM")] = None,
    config_path: Annotated[Optional[Path], typer.Option(help="Percorso config JSON")] = None,
    output: Annotated[Optional[Path], typer.Option(help="File output JSON")] = None,
    no_narrative: Annotated[bool, typer.Option(help="Salta executive summary LLM")] = False,
):
    """
    Calcola lo score da un file JSON di risposte pre-compilate (modalita' batch).

    Formato input atteso:
      {
        "intake": { "project_name": "...", ... },
        "responses": [
          { "criterion_id": "D1.C1", "score": 2, "confidence": "M", "evidence": "..." },
          { "criterion_id": "D2.C2", "not_applicable": true, "evidence": "Nessun dato personale" }
        ],
        "assessed_dimensions": ["D1","D2"]    # opzionale; default: dedotte dalle risposte
      }
    """
    _print_header()
    session = _load_session_file(input)

    try:
        cfg    = load_config(config_path)
        engine = ScoringEngine(cfg)
        session = engine.compute(session)
    except Exception as e:
        _err(f"Errore scoring: {e}")
        raise typer.Exit(1)

    typer.echo("\n" + engine.summary(session))
    _partial_warning(session)

    if not no_narrative:
        try:
            prov  = get_provider(llm, model=model)
            agent = AssessmentAgent(provider=prov, config=cfg, engine=engine)
            typer.echo("\nGenerazione executive summary...")
            typer.echo(agent.generate_executive_summary(session))
        except Exception as e:
            _warn(f"Executive summary non generato: {e}")

    out_path = output or input.with_suffix(".scored.json")
    _save_session(session, out_path)
    _ok(f"Risultati salvati in: {out_path}")


# ---------------------------------------------------------------------------
# Comando: summary
# ---------------------------------------------------------------------------

@app.command()
def summary(
    session_file: Annotated[Path, typer.Argument(help="File JSON sessione assessment")],
    config_path: Annotated[Optional[Path], typer.Option(help="Percorso config JSON")] = None,
):
    """Stampa il summary di una sessione di assessment precedentemente salvata."""
    _print_header()
    session = _load_session_file(session_file)
    try:
        engine = ScoringEngine(load_config(config_path))
        if session.composite_score is None:
            _warn("Sessione non ancora scored. Ricalcolo...")
            session = engine.compute(session)
    except Exception as e:
        _err(f"Errore: {e}")
        raise typer.Exit(1)

    typer.echo("\n" + engine.summary(session))
    _partial_warning(session)


# ---------------------------------------------------------------------------
# Comando: report
# ---------------------------------------------------------------------------

@app.command()
def report(
    session_file: Annotated[Path, typer.Argument(help="File JSON sessione assessment")],
    output:       Annotated[Optional[Path], typer.Option(help="Percorso output .md")] = None,
    llm:          Annotated[str,  typer.Option(help="Provider LLM per le narrative")] = "ollama",
    model:        Annotated[Optional[str],  typer.Option(help="Modello LLM")] = None,
    config_path:  Annotated[Optional[Path], typer.Option(help="Percorso config JSON")] = None,
    no_narrative: Annotated[bool, typer.Option(help="Salta executive summary e remediation")] = False,
):
    """
    Genera un report Markdown da una sessione di assessment salvata.

    Esempi:
      aipaf report assessment_progetto.json
      aipaf report assessment_progetto.json --output output/report.md --no-narrative
    """
    _print_header()
    session = _load_session_file(session_file)
    try:
        cfg    = load_config(config_path)
        engine = ScoringEngine(cfg)
        if session.composite_score is None:
            _warn("Sessione non scored. Ricalcolo in corso...")
            session = engine.compute(session)
    except Exception as e:
        _err(f"Errore caricamento: {e}")
        raise typer.Exit(1)

    typer.echo("\n" + engine.summary(session))
    _partial_warning(session)

    exec_summary = ""
    remediation  = ""
    if not no_narrative:
        try:
            prov  = get_provider(llm, model=model)
            agent = AssessmentAgent(provider=prov, config=cfg, engine=engine)
            typer.echo("\nGenerazione executive summary...")
            exec_summary = agent.generate_executive_summary(session)
            has_weak = any(d.counts and d.raw_score < REMEDIATION_THRESHOLD
                           for d in session.dimension_scores)
            if has_weak:
                typer.echo("Generazione remediation plan...")
                remediation = agent.generate_remediation_plan(session)
        except Exception as e:
            _warn(f"Narrative non generate: {e}")

    from aipaf.report import ReportGenerator
    rep_path = ReportGenerator().generate_and_save(session, exec_summary, remediation, output_path=output)
    _ok(f"Report salvato in: {rep_path}")


# ---------------------------------------------------------------------------
# Comandi: rag ingest / rag status
# ---------------------------------------------------------------------------

@rag_app.command("ingest")
def rag_ingest(
    force:   Annotated[bool, typer.Option(help="Forza re-ingestione anche se il file non e' cambiato")] = False,
    backend: Annotated[str,  typer.Option(help="Backend embedding: ollama | hash")] = "ollama",
    path:    Annotated[Optional[Path], typer.Option(help="Indicizza un singolo file")] = None,
    source:  Annotated[str,  typer.Option(help="Tipo sorgente: regulatory | company")] = "regulatory",
):
    """
    Indicizza i documenti in docs/sources/ nel vector store ChromaDB.

    Esempi:
      aipaf rag ingest                              # tutto
      aipaf rag ingest --force                      # forza re-ingestione
      aipaf rag ingest --path docs/sources/regulatory/eu_ai_act.pdf
      aipaf rag ingest --backend hash               # modalita' test/offline
    """
    _print_header()
    typer.echo("\nIngestione documenti RAG...\n")

    try:
        from aipaf.rag.ingestor import DocumentIngestor, DocumentSource
        ingestor = DocumentIngestor(embedding_backend=backend)
    except Exception as e:
        _err(f"Errore inizializzazione RAG: {e}")
        raise typer.Exit(1)

    if path:
        src_type = DocumentSource(source)
        try:
            doc = ingestor.ingest_file(path, src_type, force=force)
            if doc:
                _ok(f"{doc.filename} — {doc.chunk_count} chunk indicizzati")
            else:
                _warn(f"{path.name} già aggiornato (usa --force per re-indicizzare)")
        except Exception as e:
            _err(f"Errore ingestione {path}: {e}")
            raise typer.Exit(1)
    else:
        results = ingestor.ingest_all(force=force)
        if results:
            total_chunks = sum(r.chunk_count for r in results)
            _ok(f"Completato — {len(results)} file, {total_chunks} chunk totali")
        else:
            _warn("Nessun file nuovo da indicizzare. Aggiungi documenti in docs/sources/")

    status = ingestor.status()
    typer.echo(
        f"\n  Index status: "
        f"{status['regulatory']['chunks']} chunk normativi, "
        f"{status['company']['chunks']} chunk aziendali"
    )


@rag_app.command("status")
def rag_status(
    backend: Annotated[Optional[str], typer.Option(
        help="Backend embedding: ollama | hash. Default: quello persistito nell'indice"
    )] = None,
):
    """Mostra lo stato del vector store (documenti indicizzati, chunk, versioni)."""
    _print_header()

    try:
        from aipaf.rag.ingestor import DocumentIngestor
        ingestor = DocumentIngestor(embedding_backend=backend)
        status   = ingestor.status()
    except Exception as e:
        _err(f"Errore: {e}")
        raise typer.Exit(1)

    _section("RAG Index Status")
    typer.echo(f"\n  Embedding : {DocumentIngestor.persisted_backend() or 'n/d'}")
    for src_type in ("regulatory", "company"):
        s = status[src_type]
        typer.echo(f"\n  [{src_type.upper()}]")
        typer.echo(f"    Documenti : {s['documents']}")
        typer.echo(f"    Chunk     : {s['chunks']}")
        if s["files"]:
            typer.echo("    Files     :")
            for f in s["files"]:
                typer.echo(f"      · {f}")
        else:
            typer.echo("    Files     : (nessuno)")


if __name__ == "__main__":
    app()
