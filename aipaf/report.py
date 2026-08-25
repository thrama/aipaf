"""
AIPAF — AI Project Assessment Framework
report.py — Generazione report Markdown da AssessmentSession

Produce un file .md strutturato con:
  - Intestazione progetto, gate decision, copertura (completa/parziale), LLM usato
  - Scorecard con barre ASCII proporzionali agli score
  - Tabella criteri con score, confidence ed evidenza per dimensione (N/A esplicito)
  - Executive summary narrativo (generato da LLM via agent)
  - Remediation plan per le dimensioni sotto soglia (generato da LLM)
  - Appendice: pattern alert e flag normativi

Autore: Lorenzo Lombardi
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import ApprovalGate, AssessmentSession


# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------

_GATE_EMOJI = {
    ApprovalGate.GREEN:   "🟢",
    ApprovalGate.YELLOW:  "🟡",
    ApprovalGate.RED:     "🔴",
    ApprovalGate.PARTIAL: "⚪",
}

_SCORE_LABEL = {
    3: "Excellent",
    2: "Adequate",
    1: "Insufficient",
    0: "Critical",
}


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Genera report Markdown da una AssessmentSession completata.

    Usage:
        gen  = ReportGenerator()
        md   = gen.generate(session, executive_summary, remediation_plan)
        path = gen.generate_and_save(session, executive_summary, remediation_plan)
    """

    def generate(
        self,
        session:           AssessmentSession,
        executive_summary: str = "",
        remediation_plan:  str = "",
    ) -> str:
        sections: list[str] = []

        sections.append(self._header(session))
        if session.partial:
            sections.append(self._partial_notice(session))
        sections.append(self._scorecard(session))
        sections.append(self._dimension_detail(session))

        if executive_summary:
            sections.append(self._section("Executive Summary", executive_summary))

        if remediation_plan:
            sections.append(self._section("Remediation Plan", remediation_plan))

        appendix = self._appendix(session)
        if appendix:
            sections.append(appendix)

        sections.append(self._footer(session))

        return "\n\n---\n\n".join(sections)

    def save(self, markdown: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    def generate_and_save(
        self,
        session:           AssessmentSession,
        executive_summary: str = "",
        remediation_plan:  str = "",
        output_path:       Optional[Path] = None,
    ) -> Path:
        if output_path is None:
            project_slug = _slugify(
                session.intake.project_name if session.intake else "assessment"
            )
            date_str     = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
            output_path  = Path("output") / f"AIPAF_Report_{project_slug}_{date_str}.md"

        md = self.generate(session, executive_summary, remediation_plan)
        self.save(md, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Sezioni
    # ------------------------------------------------------------------

    def _header(self, session: AssessmentSession) -> str:
        intake = session.intake
        gate   = session.gate
        score  = session.composite_score or 0.0

        if gate is None:
            gate_str = "N/A"
        elif gate == ApprovalGate.PARTIAL:
            prov = session.provisional_gate.value if session.provisional_gate else "N/A"
            gate_str = f"{_GATE_EMOJI[gate]} **PARTIAL** — provvisorio sulle dimensioni valutate: {prov}"
        else:
            gate_str = f"{_GATE_EMOJI.get(gate, '')} **{gate.value}**"

        lines = [
            "# AIPAF Assessment Report",
            "",
            "| Campo | Valore |",
            "|---|---|",
        ]

        if intake:
            lines += [
                f"| **Progetto** | {_cell(intake.project_name)} |",
                f"| **Use case** | {_cell(intake.use_case_description)} |",
                f"| **Responsabile** | {_cell(intake.project_owner)} |",
                f"| **Settore** | {intake.sector.value} |",
                f"| **Stadio** | {_cell(intake.project_stage)} |",
                f"| **EU AI Act** | {intake.eu_ai_act_level.value} |",
                f"| **Data assessment** | {intake.assessment_date} |",
            ]

        coverage = "Completa" if not session.partial else (
            f"Parziale — valutate: {', '.join(session.assessed_dimensions) or 'nessuna'}"
        )
        lines += [
            f"| **Copertura** | {coverage} |",
            f"| **Composite Score** | `{score:.2f} / 3.00` |",
            f"| **Gate Decision** | {gate_str} |",
        ]
        if session.llm_provider:
            lines.append(f"| **LLM** | {session.llm_provider} / {session.llm_model or '?'} |")

        return "\n".join(lines)

    def _partial_notice(self, session: AssessmentSession) -> str:
        missing = ", ".join(session.unassessed_dimensions) or "—"
        return (
            "> **Assessment parziale.** Le dimensioni **" + missing + "** non sono state "
            "valutate in questa sessione. Composite score e gate provvisorio sono calcolati "
            "sulle sole dimensioni valutate e **non costituiscono una decisione di go/no-go**. "
            "Completare l'assessment prima della review."
        )

    def _scorecard(self, session: AssessmentSession) -> str:
        lines = [
            "## Scorecard",
            "",
            "```",
            f"{'Dimensione':<42} {'Score':>6}  Barra",
            "─" * 65,
        ]

        for d in session.dimension_scores:
            label = _fit(f"{d.dimension_id} — {d.dimension_name}", 41)
            if not d.assessed:
                lines.append(f"{label:<42} {'n/v':>6}  (non valutata)")
                continue
            if not d.applicable:
                lines.append(f"{label:<42} {'N/A':>6}  (tutti i criteri non applicabili)")
                continue
            bar      = _bar(d.raw_score, width=20)
            critical = " ⚠" if d.is_critical else ""
            lines.append(f"{label:<42} {d.raw_score:>5.2f}  {bar}{critical}")

        composite = session.composite_score or 0.0
        comp_label = "COMPOSITE SCORE" + (" (parziale)" if session.partial else "")
        lines += [
            "─" * 65,
            f"{comp_label:<42} {composite:>5.2f}  {_bar(composite, width=20)}",
            "```",
        ]

        return "\n".join(lines)

    def _dimension_detail(self, session: AssessmentSession) -> str:
        lines = ["## Dettaglio per Dimensione"]

        for d in session.dimension_scores:
            if not d.assessed:
                lines.append(f"\n### {d.dimension_id} — {d.dimension_name}  `non valutata`")
                lines.append("")
                lines.append("_Dimensione esclusa da questa sessione di assessment._")
                continue

            if not d.applicable:
                lines.append(f"\n### {d.dimension_id} — {d.dimension_name}  `N/A`")
                lines.append("")
                lines.append("_Tutti i criteri sono stati marcati come non applicabili; "
                             "la dimensione non concorre a composito e gate._")
                self._append_na_notes(lines, d)
                continue

            critical_label = "  ⚠️ CRITICO" if d.is_critical else ""
            lines.append(
                f"\n### {d.dimension_id} — {d.dimension_name}"
                f"  `{d.raw_score:.2f}/3.00`{critical_label}"
            )
            lines.append("")

            if not d.criteria_responses:
                lines.append("_Nessuna risposta registrata per questa dimensione._")
                continue

            lines += [
                "| Criterio | Score | Conf. | Evidenza |",
                "|---|:---:|:---:|---|",
            ]

            for r in d.criteria_responses:
                if r.not_applicable:
                    lines.append(
                        f"| `{r.criterion_id}` | N/A | — | {_cell(r.evidence or 'Non applicabile')} |"
                    )
                    continue
                score_label = _SCORE_LABEL.get(r.score, str(r.score))
                lines.append(
                    f"| `{r.criterion_id}` "
                    f"| {r.score} — {score_label} "
                    f"| {r.confidence.value} "
                    f"| {_cell(r.evidence[:120] if r.evidence else '—')} |"
                )

            if d.flags:
                lines += ["", "**Flag attivi:**"]
                for flag in d.flags:
                    lines.append(f"- {flag}")

            if d.data_needed:
                lines.append("")
                lines.append(
                    "**Data needed** (confidence L): "
                    + ", ".join(f"`{c}`" for c in d.data_needed)
                )

            if d.not_applicable:
                lines.append("")
                lines.append(
                    "**Non applicabili** (esclusi dal denominatore): "
                    + ", ".join(f"`{c}`" for c in d.not_applicable)
                )

        return "\n".join(lines)

    @staticmethod
    def _append_na_notes(lines: list[str], d) -> None:
        notes = [(r.criterion_id, r.evidence) for r in d.criteria_responses if r.not_applicable]
        if notes:
            lines.append("")
            for cid, ev in notes:
                lines.append(f"- `{cid}`: {ev or 'Non applicabile'}")

    def _section(self, title: str, content: str) -> str:
        return f"## {title}\n\n{content.strip()}"

    def _appendix(self, session: AssessmentSession) -> str:
        parts: list[str] = []

        active_patterns = [p for p in session.pattern_alerts if p.triggered]
        if active_patterns:
            lines = ["## Appendice — Pattern Alert", ""]
            for p in active_patterns:
                lines += [
                    f"### ⚡ {p.label} — {p.name}",
                    "",
                    p.description,
                    "",
                    f"**Azione raccomandata:** {p.recommended_action}",
                    "",
                ]
            parts.append("\n".join(lines))

        if session.normative_red_flags:
            lines = [
                "## Appendice — Red Flag Normativi",
                "",
                "| Riferimento | Criterio | Descrizione |",
                "|---|---|---|",
            ]
            for f in session.normative_red_flags:
                lines.append(f"| **{f.norm_reference}** | `{f.criterion_id}` | {_cell(f.description)} |")
            parts.append("\n".join(lines))

        return "\n\n".join(parts)

    def _footer(self, session: AssessmentSession) -> str:
        generated_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        llm = (
            f"{session.llm_provider}/{session.llm_model or '?'}"
            if session.llm_provider else "nessuno (scoring deterministico)"
        )
        version = session.aipaf_version or "?"
        return (
            f"---\n\n"
            f"_Report generato da AIPAF Agent v{version} il {generated_at}._  \n"
            f"_Interpretazione risposte: {llm}._  \n"
            f"_Framework: NIST AI RMF 1.0 · ISO/IEC 42001:2023 · EU AI Act._"
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _cell(text: str) -> str:
    """Rende sicuro un testo per una cella di tabella Markdown."""
    return (text or "").replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _fit(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 1] + "…"


def _bar(score: float, width: int = 20) -> str:
    filled = int(round(score / 3.0 * width))
    return "█" * filled + "░" * (width - filled)


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:40]
