"""
AIPAF — Test suite per report.py
Eseguire: python -m pytest tests/test_report.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from aipaf.models import (
    ApprovalGate, AssessmentSession, ConfidenceLevel, CriterionResponse,
    EUAIActLevel, NormativeRedFlag, PatternAlert, ProjectIntake, Sector,
)
from aipaf.engine import ScoringEngine, load_config
from aipaf.report import ReportGenerator, _bar, _cell, _slugify


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    return ScoringEngine(load_config())


def make_session(score: int = 2, confidence: ConfidenceLevel = ConfidenceLevel.HIGH,
                 eu_level: EUAIActLevel = EUAIActLevel.MINIMAL_RISK) -> AssessmentSession:
    session = AssessmentSession(intake=ProjectIntake(
        project_name="Progetto Test",
        use_case_description="Classificatore documenti interni",
        project_owner="Lorenzo Lombardi",
        sector=Sector.DEFAULT,
        eu_ai_act_level=eu_level,
        project_stage="PoC",
        assessment_date="2026-04-28",
    ))
    for cid in [
        "D1.C1","D1.C2","D1.C3","D1.C4","D1.C5",
        "D2.C1","D2.C2","D2.C3","D2.C4","D2.C5",
        "D3.C1","D3.C2","D3.C3","D3.C4","D3.C5",
        "D4.C1","D4.C2","D4.C3","D4.C4","D4.C5","D4.C6",
        "D5.C1","D5.C2","D5.C3","D5.C4","D5.C5","D5.C6",
        "D6.C1","D6.C2","D6.C3","D6.C4","D6.C5",
    ]:
        session.upsert_response(CriterionResponse(
            criterion_id=cid, score=score, confidence=confidence,
            evidence="Evidenza di test per questo criterio."
        ))
    return session


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_bar_full(self):
        assert _bar(3.0, width=10) == "██████████"

    def test_bar_empty(self):
        assert _bar(0.0, width=10) == "░░░░░░░░░░"

    def test_bar_half(self):
        b = _bar(1.5, width=10)
        assert len(b) == 10
        assert "█" in b and "░" in b

    def test_slugify_basic(self):
        assert _slugify("Progetto Test") == "progetto_test"

    def test_slugify_special_chars(self):
        slug = _slugify("Progetto AI & Governance!")
        assert " " not in slug
        assert "&" not in slug

    def test_slugify_max_length(self):
        assert len(_slugify("a" * 100)) <= 40


# ---------------------------------------------------------------------------
# ReportGenerator — struttura Markdown
# ---------------------------------------------------------------------------

class TestReportStructure:

    def test_generate_returns_string(self, engine):
        session = engine.compute(make_session())
        md = ReportGenerator().generate(session)
        assert isinstance(md, str)
        assert len(md) > 100

    def test_header_present(self, engine):
        session = engine.compute(make_session())
        md = ReportGenerator().generate(session)
        assert "# AIPAF Assessment Report" in md
        assert "Progetto Test" in md
        assert "Lorenzo Lombardi" in md

    def test_gate_in_header(self, engine):
        session = engine.compute(make_session(score=3))
        md = ReportGenerator().generate(session)
        assert "GREEN" in md or "YELLOW" in md or "RED" in md

    def test_scorecard_present(self, engine):
        session = engine.compute(make_session())
        md = ReportGenerator().generate(session)
        assert "## Scorecard" in md
        assert "COMPOSITE SCORE" in md

    def test_all_dimensions_in_detail(self, engine):
        session = engine.compute(make_session())
        md = ReportGenerator().generate(session)
        for dim in ("D1", "D2", "D3", "D4", "D5", "D6"):
            assert dim in md

    def test_executive_summary_section(self, engine):
        session = engine.compute(make_session())
        md = ReportGenerator().generate(session, executive_summary="Questo è il summary.")
        assert "## Executive Summary" in md
        assert "Questo è il summary." in md

    def test_no_executive_summary_if_empty(self, engine):
        session = engine.compute(make_session())
        md = ReportGenerator().generate(session, executive_summary="")
        assert "## Executive Summary" not in md

    def test_remediation_section(self, engine):
        session = engine.compute(make_session())
        md = ReportGenerator().generate(session, remediation_plan="Azione 1: fare X.")
        assert "## Remediation Plan" in md
        assert "Azione 1: fare X." in md

    def test_footer_present(self, engine):
        session = engine.compute(make_session())
        md = ReportGenerator().generate(session)
        assert "NIST AI RMF" in md
        assert "EU AI Act" in md

    def test_normative_flags_in_appendix(self, engine):
        session = engine.compute(make_session(
            score=1, eu_level=EUAIActLevel.HIGH_RISK
        ))
        md = ReportGenerator().generate(session)
        assert "Art. 14 EU AI Act" in md or "Art. 13 EU AI Act" in md

    def test_pattern_alert_in_appendix(self, engine):
        """Pattern PA3 (Zillow): D4 alta, D6 bassa."""
        session = make_session(score=3)
        for cid in ["D6.C1","D6.C2","D6.C3","D6.C4","D6.C5"]:
            session.upsert_response(CriterionResponse(
                criterion_id=cid, score=0, confidence=ConfidenceLevel.HIGH
            ))
        session = engine.compute(session)
        md = ReportGenerator().generate(session)
        assert "Pattern Alert" in md


# ---------------------------------------------------------------------------
# ReportGenerator — salvataggio su file
# ---------------------------------------------------------------------------

class TestReportSave:

    def test_save_creates_file(self, tmp_path, engine):
        session  = engine.compute(make_session())
        gen      = ReportGenerator()
        md       = gen.generate(session)
        out_path = tmp_path / "report.md"
        gen.save(md, out_path)
        assert out_path.exists()
        assert out_path.read_text(encoding="utf-8") == md

    def test_save_creates_parent_dirs(self, tmp_path, engine):
        session  = engine.compute(make_session())
        gen      = ReportGenerator()
        md       = gen.generate(session)
        out_path = tmp_path / "nested" / "dir" / "report.md"
        gen.save(md, out_path)
        assert out_path.exists()

    def test_generate_and_save_default_path(self, tmp_path, engine, monkeypatch):
        """generate_and_save crea il file in output/ con nome automatico."""
        monkeypatch.chdir(tmp_path)
        session  = engine.compute(make_session())
        gen      = ReportGenerator()
        rep_path = gen.generate_and_save(session)
        assert rep_path.exists()
        assert rep_path.suffix == ".md"
        assert "progetto_test" in rep_path.name.lower()

    def test_generate_and_save_explicit_path(self, tmp_path, engine):
        session  = engine.compute(make_session())
        gen      = ReportGenerator()
        out_path = tmp_path / "custom_report.md"
        rep_path = gen.generate_and_save(session, output_path=out_path)
        assert rep_path == out_path
        assert out_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# v0.2.0: PARTIAL, N/A, tracciabilita', robustezza tabelle
# ---------------------------------------------------------------------------

class TestReportV020:

    def test_partial_report(self, engine):
        session = make_session()
        session.responses = [r for r in session.responses if r.criterion_id.startswith(("D1.", "D2."))]
        session = engine.compute(session, assessed_dimensions=["D1", "D2"])
        md = ReportGenerator().generate(session)
        assert "**PARTIAL**" in md
        assert "Assessment parziale" in md
        assert "D3, D4, D5, D6" in md
        assert "non valutata" in md
        assert "COMPOSITE SCORE (parziale)" in md
        assert "CRITICO" not in md          # le dimensioni assenti non sono critiche

    def test_na_criterion_rendered(self, engine):
        session = make_session()
        session.upsert_response(CriterionResponse(
            criterion_id="D2.C2", not_applicable=True, evidence="Nessun dato personale trattato"))
        session = engine.compute(session)
        md = ReportGenerator().generate(session)
        assert "| `D2.C2` | N/A | — | Nessun dato personale trattato |" in md
        assert "**Non applicabili** (esclusi dal denominatore): `D2.C2`" in md

    def test_all_na_dimension_rendered(self, engine):
        session = make_session()
        for cid in ["D5.C1","D5.C2","D5.C3","D5.C4","D5.C5","D5.C6"]:
            session.upsert_response(CriterionResponse(criterion_id=cid, not_applicable=True))
        session = engine.compute(session)
        md = ReportGenerator().generate(session)
        assert "### D5 — Ethical & Responsible AI  `N/A`" in md
        assert "tutti i criteri non applicabili" in md

    def test_llm_metadata_in_header_and_footer(self, engine):
        session = engine.compute(make_session())
        session.llm_provider = "ollama"
        session.llm_model = "qwen3.5:4b"
        session.aipaf_version = "0.2.0"
        md = ReportGenerator().generate(session)
        assert "| **LLM** | ollama / qwen3.5:4b |" in md
        assert "Interpretazione risposte: ollama/qwen3.5:4b" in md
        assert "AIPAF Agent v0.2.0" in md

    def test_no_llm_footer(self, engine):
        md = ReportGenerator().generate(engine.compute(make_session()))
        assert "scoring deterministico" in md

    def test_multiline_evidence_does_not_break_table(self, engine):
        session = make_session()
        session.upsert_response(CriterionResponse(
            criterion_id="D1.C1", score=2, confidence=ConfidenceLevel.HIGH,
            evidence="riga uno\nriga due | con pipe"))
        session = engine.compute(session)
        md = ReportGenerator().generate(session)
        row = next(l for l in md.splitlines() if l.startswith("| `D1.C1`"))
        assert "riga uno riga due \\| con pipe" in row

    def test_cell_helper(self):
        assert _cell("a\nb|c\r\n") == "a b\\|c  "
