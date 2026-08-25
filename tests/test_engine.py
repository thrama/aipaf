"""
AIPAF — Test suite per engine.py e models.py
Eseguire: python -m pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from aipaf.models import (
    DIMENSION_CRITICAL_THRESHOLD, DimensionScore,
    ApprovalGate, AssessmentSession, ConfidenceLevel, CriterionResponse,
    EUAIActLevel, ProjectIntake, Sector,
)
from aipaf.engine import ScoringEngine, evaluate_skip_condition, load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config():
    return load_config()

@pytest.fixture(scope="module")
def engine(config):
    return ScoringEngine(config)

def make_intake(**kwargs) -> ProjectIntake:
    defaults = dict(
        project_name="Test Project",
        use_case_description="Classificazione documenti",
        project_owner="Lorenzo Lombardi",
        sector=Sector.DEFAULT,
        eu_ai_act_level=EUAIActLevel.MINIMAL_RISK,
        assessment_date="2026-04-28",
    )
    defaults.update(kwargs)
    return ProjectIntake(**defaults)

ALL_CRITERIA = [
    "D1.C1","D1.C2","D1.C3","D1.C4","D1.C5",
    "D2.C1","D2.C2","D2.C3","D2.C4","D2.C5",
    "D3.C1","D3.C2","D3.C3","D3.C4","D3.C5",
    "D4.C1","D4.C2","D4.C3","D4.C4","D4.C5","D4.C6",
    "D5.C1","D5.C2","D5.C3","D5.C4","D5.C5","D5.C6",
    "D6.C1","D6.C2","D6.C3","D6.C4","D6.C5",
]

def full_session(score: int, confidence: ConfidenceLevel, **intake_kwargs) -> AssessmentSession:
    session = AssessmentSession(intake=make_intake(**intake_kwargs))
    for cid in ALL_CRITERIA:
        session.upsert_response(CriterionResponse(
            criterion_id=cid, score=score, confidence=confidence, evidence="Test"
        ))
    return session


# ---------------------------------------------------------------------------
# CriterionResponse
# ---------------------------------------------------------------------------

class TestCriterionResponse:

    def test_effective_high(self):
        r = CriterionResponse(criterion_id="D1.C1", score=3, confidence=ConfidenceLevel.HIGH)
        assert r.effective_score == 3.0

    def test_effective_medium(self):
        r = CriterionResponse(criterion_id="D1.C1", score=2, confidence=ConfidenceLevel.MEDIUM)
        assert r.effective_score == pytest.approx(1.6)

    def test_effective_low_zero(self):
        r = CriterionResponse(criterion_id="D1.C1", score=3, confidence=ConfidenceLevel.LOW)
        assert r.effective_score == 0.0
        assert r.data_needed_flag is True

    def test_invalid_score(self):
        with pytest.raises(Exception):
            CriterionResponse(criterion_id="D1.C1", score=4)


# ---------------------------------------------------------------------------
# Gate logic
# ---------------------------------------------------------------------------

class TestGateLogic:

    def test_green_all_excellent(self, engine):
        result = engine.compute(full_session(3, ConfidenceLevel.HIGH))
        assert result.gate == ApprovalGate.GREEN
        assert result.composite_score == pytest.approx(3.0, abs=0.01)

    def test_yellow_mid_scores(self, engine):
        # 2 * 0.8 = 1.6 effective -> YELLOW
        result = engine.compute(full_session(2, ConfidenceLevel.MEDIUM))
        assert result.gate == ApprovalGate.YELLOW

    def test_red_zero_scores(self, engine):
        result = engine.compute(full_session(0, ConfidenceLevel.HIGH))
        assert result.gate == ApprovalGate.RED

    def test_red_one_critical_dimension(self, engine):
        """Una dimensione < 0.5 forza RED indipendentemente dal composito."""
        session = full_session(3, ConfidenceLevel.HIGH)
        for cid in ["D1.C1","D1.C2","D1.C3","D1.C4","D1.C5"]:
            session.upsert_response(CriterionResponse(
                criterion_id=cid, score=0, confidence=ConfidenceLevel.HIGH
            ))
        result = engine.compute(session)
        assert result.gate == ApprovalGate.RED

    def test_empty_session_is_partial(self, engine):
        """Nessuna risposta = nessuna dimensione valutata: gate PARTIAL, non RED."""
        result = engine.compute(AssessmentSession(intake=make_intake()))
        assert result.composite_score == 0.0
        assert result.partial is True
        assert result.gate == ApprovalGate.PARTIAL
        assert result.assessed_dimensions == []
        assert all(not d.assessed for d in result.dimension_scores)

    def test_threshold_single_source(self, engine):
        """DimensionScore.is_critical ed engine usano la stessa costante."""
        d = DimensionScore(dimension_id="D1", dimension_name="x", weight_pct=20,
                           raw_score=DIMENSION_CRITICAL_THRESHOLD - 0.01)
        assert d.is_critical is True
        d.raw_score = DIMENSION_CRITICAL_THRESHOLD
        assert d.is_critical is False

    def test_unknown_criterion_raises(self, engine):
        session = full_session(3, ConfidenceLevel.HIGH)
        session.responses.append(CriterionResponse(criterion_id="D3.C9", score=3))
        with pytest.raises(ValueError, match="D3.C9"):
            engine.compute(session)


# ---------------------------------------------------------------------------
# Assessment parziale
# ---------------------------------------------------------------------------

class TestPartialAssessment:

    def _partial(self, engine, dims, score=3, conf=ConfidenceLevel.HIGH):
        session = AssessmentSession(intake=make_intake())
        for cid in ALL_CRITERIA:
            if cid.split(".")[0] in dims:
                session.upsert_response(CriterionResponse(
                    criterion_id=cid, score=score, confidence=conf, evidence="Test"))
        return engine.compute(session, assessed_dimensions=dims)

    def test_partial_gate_and_provisional(self, engine):
        result = self._partial(engine, ["D1", "D2"])
        assert result.partial is True
        assert result.gate == ApprovalGate.PARTIAL
        assert result.provisional_gate == ApprovalGate.GREEN
        assert result.assessed_dimensions == ["D1", "D2"]
        assert result.unassessed_dimensions == ["D3", "D4", "D5", "D6"]

    def test_partial_composite_only_on_assessed(self, engine):
        result = self._partial(engine, ["D1", "D2"])
        # D1=D2=3.0 → composito 3.0, non schiacciato dalle dimensioni assenti
        assert result.composite_score == pytest.approx(3.0, abs=0.01)

    def test_unassessed_not_critical(self, engine):
        result = self._partial(engine, ["D1"])
        d5 = next(d for d in result.dimension_scores if d.dimension_id == "D5")
        assert d5.assessed is False
        assert d5.is_critical is False

    def test_no_pattern_alert_from_unassessed(self, engine):
        """PA1 (D2 alto, D5 basso) non deve scattare se D5 non e' valutata."""
        result = self._partial(engine, ["D2"])
        pa1 = next(p for p in result.pattern_alerts if p.pattern_id == "PA1")
        assert pa1.triggered is False

    def test_no_normative_flag_from_unassessed(self, engine):
        session = AssessmentSession(intake=make_intake(eu_ai_act_level=EUAIActLevel.HIGH_RISK))
        for cid in ["D1.C1","D1.C2","D1.C3","D1.C4","D1.C5"]:
            session.upsert_response(CriterionResponse(criterion_id=cid, score=3, confidence=ConfidenceLevel.HIGH))
        result = engine.compute(session, assessed_dimensions=["D1"])
        assert result.normative_red_flags == []   # D5.C3/D4.C4 non valutati → nessun flag

    def test_inferred_from_responses(self, engine):
        """Senza assessed_dimensions esplicite, le dimensioni si deducono dalle risposte."""
        session = AssessmentSession(intake=make_intake())
        for cid in ["D3.C1","D3.C2","D3.C3","D3.C4","D3.C5"]:
            session.upsert_response(CriterionResponse(criterion_id=cid, score=2, confidence=ConfidenceLevel.HIGH))
        result = engine.compute(session)
        assert result.assessed_dimensions == ["D3"]
        assert result.gate == ApprovalGate.PARTIAL

    def test_full_session_not_partial(self, engine):
        result = engine.compute(full_session(2, ConfidenceLevel.HIGH))
        assert result.partial is False
        assert result.provisional_gate is None
        assert result.gate == ApprovalGate.GREEN


# ---------------------------------------------------------------------------
# Criteri N/A
# ---------------------------------------------------------------------------

class TestNotApplicable:

    def test_na_excluded_from_denominator(self, engine):
        """D2.C2 (25%) N/A: D2 = media pesata sui restanti 75%."""
        session = full_session(3, ConfidenceLevel.HIGH)
        session.upsert_response(CriterionResponse(criterion_id="D2.C2", not_applicable=True,
                                                  evidence="Nessun dato personale"))
        # abbasso D2.C1 (30%) a 0 per verificare la rinormalizzazione
        session.upsert_response(CriterionResponse(criterion_id="D2.C1", score=0, confidence=ConfidenceLevel.HIGH))
        result = engine.compute(session)
        d2 = next(d for d in result.dimension_scores if d.dimension_id == "D2")
        # (0*30 + 3*20 + 3*15 + 3*10) / 75 = 135/75 = 1.8
        assert d2.raw_score == pytest.approx(1.8, abs=0.001)
        assert d2.not_applicable == ["D2.C2"]
        assert "D2.C2" not in d2.data_needed

    def test_na_does_not_penalize(self, engine):
        session = full_session(3, ConfidenceLevel.HIGH)
        session.upsert_response(CriterionResponse(criterion_id="D2.C2", not_applicable=True))
        session.upsert_response(CriterionResponse(criterion_id="D3.C4", not_applicable=True))
        result = engine.compute(session)
        assert result.composite_score == pytest.approx(3.0, abs=0.01)
        assert result.gate == ApprovalGate.GREEN

    def test_all_na_dimension_excluded(self, engine):
        session = full_session(3, ConfidenceLevel.HIGH)
        for cid in ["D5.C1","D5.C2","D5.C3","D5.C4","D5.C5","D5.C6"]:
            session.upsert_response(CriterionResponse(criterion_id=cid, not_applicable=True))
        result = engine.compute(session)
        d5 = next(d for d in result.dimension_scores if d.dimension_id == "D5")
        assert d5.assessed is True
        assert d5.applicable is False
        assert d5.is_critical is False
        assert result.not_applicable_dimensions == ["D5"]
        assert result.gate == ApprovalGate.GREEN
        assert result.partial is False

    def test_na_response_effective_zero_and_no_data_needed(self):
        r = CriterionResponse(criterion_id="D2.C2", score=3, confidence=ConfidenceLevel.HIGH,
                              not_applicable=True)
        assert r.effective_score == 0.0
        assert r.data_needed_flag is False

    def test_na_excluded_from_normative_flags(self, engine):
        session = full_session(0, ConfidenceLevel.HIGH, eu_ai_act_level=EUAIActLevel.HIGH_RISK)
        session.upsert_response(CriterionResponse(criterion_id="D5.C3", not_applicable=True))
        result = engine.compute(session)
        assert "D5.C3" not in [f.criterion_id for f in result.normative_red_flags]
        assert "D4.C4" in [f.criterion_id for f in result.normative_red_flags]


# ---------------------------------------------------------------------------
# skip_condition
# ---------------------------------------------------------------------------

class TestSkipCondition:

    def test_true_when_no_personal_data(self):
        intake = make_intake(personal_data_treated=False)
        assert evaluate_skip_condition("personal_data_treated == false", intake) is True

    def test_false_when_personal_data(self):
        intake = make_intake(personal_data_treated=True)
        assert evaluate_skip_condition("personal_data_treated == false", intake) is False

    def test_not_equal_and_enum(self):
        intake = make_intake(sector=Sector.HEALTHCARE)
        assert evaluate_skip_condition("sector != 'healthcare'", intake) is False
        assert evaluate_skip_condition("sector == healthcare", intake) is True

    def test_none_condition(self):
        assert evaluate_skip_condition(None, make_intake()) is False

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError, match="P0.7"):
            evaluate_skip_condition("P0.7 == 'no_personal_data'", make_intake())

    def test_config_conditions_are_valid(self, engine):
        """Tutte le skip_condition del config devono essere valutabili."""
        intake = make_intake(personal_data_treated=False)
        candidates = engine.candidate_not_applicable(intake)
        assert set(candidates) == {"D2.C2", "D3.C4"}
        assert engine.candidate_not_applicable(make_intake(personal_data_treated=True)) == []


# ---------------------------------------------------------------------------
# Sector weights
# ---------------------------------------------------------------------------

class TestSectorWeights:

    def test_banking_d2_heavier(self, engine):
        """In banking D2=0 penalizza di più il composito rispetto a default."""
        def with_d2_zero(sector):
            s = full_session(3, ConfidenceLevel.HIGH, sector=sector)
            for cid in ["D2.C1","D2.C2","D2.C3","D2.C4","D2.C5"]:
                s.upsert_response(CriterionResponse(
                    criterion_id=cid, score=0, confidence=ConfidenceLevel.HIGH
                ))
            return s

        r_default = engine.compute(with_d2_zero(Sector.DEFAULT))
        r_banking  = engine.compute(with_d2_zero(Sector.BANKING_FINANCE))
        assert r_banking.composite_score < r_default.composite_score


# ---------------------------------------------------------------------------
# Pattern alerts
# ---------------------------------------------------------------------------

class TestPatternAlerts:

    def test_zillow_pattern(self, engine):
        """PA3: D4 alta, D6 bassa."""
        s = full_session(3, ConfidenceLevel.HIGH)
        for cid in ["D6.C1","D6.C2","D6.C3","D6.C4","D6.C5"]:
            s.upsert_response(CriterionResponse(
                criterion_id=cid, score=0, confidence=ConfidenceLevel.HIGH
            ))
        result = engine.compute(s)
        pa3 = next(p for p in result.pattern_alerts if p.pattern_id == "PA3")
        assert pa3.triggered is True

    def test_no_false_positives(self, engine):
        result = engine.compute(full_session(3, ConfidenceLevel.HIGH))
        assert all(not p.triggered for p in result.pattern_alerts)


# ---------------------------------------------------------------------------
# Normative red flags
# ---------------------------------------------------------------------------

class TestNormativeFlags:

    def test_high_risk_hitl(self, engine):
        s = full_session(3, ConfidenceLevel.HIGH, eu_ai_act_level=EUAIActLevel.HIGH_RISK)
        s.upsert_response(CriterionResponse(
            criterion_id="D5.C3", score=1, confidence=ConfidenceLevel.HIGH
        ))
        result = engine.compute(s)
        assert "Art. 14 EU AI Act" in [f.norm_reference for f in result.normative_red_flags]

    def test_minimal_risk_no_flags(self, engine):
        result = engine.compute(full_session(3, ConfidenceLevel.HIGH))
        assert len(result.normative_red_flags) == 0

    def test_limited_risk_transparency(self, engine):
        s = full_session(3, ConfidenceLevel.HIGH, eu_ai_act_level=EUAIActLevel.LIMITED_RISK)
        s.upsert_response(CriterionResponse(
            criterion_id="D5.C2", score=1, confidence=ConfidenceLevel.HIGH
        ))
        result = engine.compute(s)
        assert "Art. 50 EU AI Act" in [f.norm_reference for f in result.normative_red_flags]


# ---------------------------------------------------------------------------
# AssessmentSession utility
# ---------------------------------------------------------------------------

class TestAssessmentSession:

    def test_upsert_overwrites(self):
        s = AssessmentSession()
        s.upsert_response(CriterionResponse(criterion_id="D1.C1", score=1, confidence=ConfidenceLevel.HIGH))
        s.upsert_response(CriterionResponse(criterion_id="D1.C1", score=3, confidence=ConfidenceLevel.HIGH))
        assert len(s.responses) == 1
        assert s.get_response("D1.C1").score == 3

    def test_is_complete_partial(self):
        s = AssessmentSession()
        s.upsert_response(CriterionResponse(criterion_id="D1.C1", score=2, confidence=ConfidenceLevel.HIGH))
        assert s.is_complete is False

    def test_summary_contains_gate(self, engine):
        result = engine.compute(full_session(2, ConfidenceLevel.HIGH))
        summary = engine.summary(result)
        assert any(g in summary for g in ("GREEN", "YELLOW", "RED"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
