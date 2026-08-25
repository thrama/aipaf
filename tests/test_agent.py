"""
AIPAF — Test suite per agent.py (provider mock, nessuna chiamata reale)

Verifica i principi di governance della v0.2.0:
  - fallimenti LLM rumorosi (LLMOutputError dopo retry)
  - chiamate strutturate stateless
  - validazione dei valori fuori dominio
Eseguire: python -m pytest tests/test_agent.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pytest

from aipaf.agent import (
    STRUCTURED_RETRIES, AssessmentAgent, LLMOutputError, _validate_criterion, _validate_intake,
)
from aipaf.engine import ScoringEngine, load_config
from aipaf.llm.base import LLMProvider, LLMResponse, Message
from aipaf.models import (
    ApprovalGate, AssessmentSession, ConfidenceLevel, EUAIActLevel, Sector,
)


# ---------------------------------------------------------------------------
# Provider mock con script di risposte
# ---------------------------------------------------------------------------

class ScriptedProvider(LLMProvider):
    """Restituisce le risposte in sequenza e registra ogni chiamata."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "scripted-1"

    def complete(self, messages, system=None, *, max_tokens=2048, temperature=0.2):
        self.calls.append({"messages": list(messages), "system": system})
        if not self._responses:
            raise AssertionError("ScriptedProvider: risposte esaurite")
        text = self._responses.pop(0)
        return LLMResponse(content=text, model=self.model_name, provider="mock",
                           input_tokens=10, output_tokens=5)


GOOD_INTAKE = json.dumps({
    "project_name": "Triage Bot",
    "use_case_description": "Chatbot di triage clienti",
    "project_owner": "Mario Rossi",
    "sector": "banking_finance",
    "ai_system_type": "chatbot",
    "eu_ai_act_level": "LIMITED_RISK",
    "project_stage": "PoC",
    "user_count_estimate": 1500,
    "vulnerable_subjects": False,
    "personal_data_treated": True,
    "sensitive_data_treated": False,
})

GOOD_CRITERION = json.dumps({
    "score": 2, "confidence": "M",
    "evidence": "Business case verbale, KPI da validare",
    "reasoning": "Problema chiaro ma metriche non formalizzate",
})


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def engine(config):
    return ScoringEngine(config)


def make_agent(responses, config, engine):
    prov = ScriptedProvider(responses)
    return AssessmentAgent(provider=prov, config=config, engine=engine), prov


# ---------------------------------------------------------------------------
# Intake
# ---------------------------------------------------------------------------

class TestIntake:

    def test_success(self, config, engine):
        agent, prov = make_agent([GOOD_INTAKE], config, engine)
        session = agent.collect_intake(AssessmentSession(), "Un chatbot bancario...")
        i = session.intake
        assert i.project_name == "Triage Bot"
        assert i.sector == Sector.BANKING_FINANCE
        assert i.eu_ai_act_level == EUAIActLevel.LIMITED_RISK
        assert i.user_count_estimate == 1500
        assert i.personal_data_treated is True
        assert len(prov.calls) == 1

    def test_intake_is_stateless(self, config, engine):
        agent, prov = make_agent([GOOD_INTAKE], config, engine)
        agent.collect_intake(AssessmentSession(), "descrizione")
        assert len(prov.calls[0]["messages"]) == 1     # nessuno storico pregresso
        assert agent._history == []                     # e nessuno storico lasciato

    def test_bad_sector_raises_after_retry(self, config, engine):
        bad = json.loads(GOOD_INTAKE); bad["sector"] = "insurance"
        agent, prov = make_agent([json.dumps(bad)] * (1 + STRUCTURED_RETRIES), config, engine)
        with pytest.raises(LLMOutputError, match="sector"):
            agent.collect_intake(AssessmentSession(), "x")
        assert len(prov.calls) == 1 + STRUCTURED_RETRIES

    def test_bad_eu_level_raises(self, config, engine):
        bad = json.loads(GOOD_INTAKE); bad["eu_ai_act_level"] = "MEDIUM"
        agent, _ = make_agent([json.dumps(bad)] * (1 + STRUCTURED_RETRIES), config, engine)
        with pytest.raises(LLMOutputError, match="eu_ai_act_level"):
            agent.collect_intake(AssessmentSession(), "x")

    def test_retry_recovers(self, config, engine):
        """Primo output rotto, secondo valido: nessun errore, intake corretto."""
        agent, prov = make_agent(["ecco:  {non json", GOOD_INTAKE], config, engine)
        session = agent.collect_intake(AssessmentSession(), "x")
        assert session.intake.project_name == "Triage Bot"
        assert len(prov.calls) == 2
        assert "ATTENZIONE" in prov.calls[1]["messages"][0].content

    def test_no_silent_default_on_garbage(self, config, engine):
        """Il caso che in v0.1 produceva sector=default / MINIMAL_RISK."""
        agent, _ = make_agent(["Non posso aiutarti."] * (1 + STRUCTURED_RETRIES), config, engine)
        with pytest.raises(LLMOutputError):
            agent.collect_intake(AssessmentSession(), "x")

    def test_case_insensitive_enums(self):
        d = json.loads(GOOD_INTAKE)
        d["sector"] = "Banking_Finance"; d["eu_ai_act_level"] = "limited_risk"
        out = _validate_intake(d)
        assert out["sector"] == Sector.BANKING_FINANCE
        assert out["eu_ai_act_level"] == EUAIActLevel.LIMITED_RISK

    def test_missing_project_name_raises(self):
        d = json.loads(GOOD_INTAKE); d["project_name"] = ""
        with pytest.raises(LLMOutputError, match="project_name"):
            _validate_intake(d)


# ---------------------------------------------------------------------------
# Interpretazione criteri
# ---------------------------------------------------------------------------

class TestCriterion:

    def test_success(self, config, engine):
        agent, _ = make_agent([GOOD_CRITERION], config, engine)
        r = agent.interpret_criterion_response("D1.C1", "Abbiamo un problema chiaro", config.dimension_map["D1"])
        assert r.criterion_id == "D1.C1"
        assert r.score == 2
        assert r.confidence == ConfidenceLevel.MEDIUM
        assert r.not_applicable is False
        assert "verbale" in r.evidence

    def test_criteria_are_stateless(self, config, engine):
        """La seconda chiamata non deve contenere la prima nel contesto."""
        agent, prov = make_agent([GOOD_CRITERION, GOOD_CRITERION], config, engine)
        d1 = config.dimension_map["D1"]
        agent.interpret_criterion_response("D1.C1", "risposta uno", d1)
        agent.interpret_criterion_response("D1.C2", "risposta due", d1)
        assert len(prov.calls[1]["messages"]) == 1
        assert "risposta uno" not in prov.calls[1]["messages"][0].content
        assert "D1.C2" in prov.calls[1]["messages"][0].content

    def test_score_out_of_range_raises(self, config, engine):
        bad = json.dumps({"score": 5, "confidence": "H", "evidence": "", "reasoning": ""})
        agent, _ = make_agent([bad] * (1 + STRUCTURED_RETRIES), config, engine)
        with pytest.raises(LLMOutputError, match="score"):
            agent.interpret_criterion_response("D1.C1", "x", config.dimension_map["D1"])

    def test_bad_confidence_raises(self):
        with pytest.raises(LLMOutputError, match="confidence"):
            _validate_criterion({"score": 2, "confidence": "HIGH"})

    def test_markdown_fenced_json_ok(self, config, engine):
        agent, _ = make_agent(["```json\n" + GOOD_CRITERION + "\n```"], config, engine)
        r = agent.interpret_criterion_response("D1.C1", "x", config.dimension_map["D1"])
        assert r.score == 2

    def test_empty_evidence_falls_back_to_answer(self, config, engine):
        resp = json.dumps({"score": 1, "confidence": "L", "evidence": "", "reasoning": "n/d"})
        agent, _ = make_agent([resp], config, engine)
        r = agent.interpret_criterion_response("D1.C1", "testo dell'assessor", config.dimension_map["D1"])
        assert r.evidence == "testo dell'assessor"


# ---------------------------------------------------------------------------
# Finalize e narrative
# ---------------------------------------------------------------------------

class TestFinalizeAndNarratives:

    def _scored(self, config, engine, responses):
        agent, prov = make_agent(responses, config, engine)
        session = AssessmentSession()
        session = agent.collect_intake(session, "x")
        d1 = config.dimension_map["D1"]
        for c in d1.criteria:
            session.upsert_response(agent.interpret_criterion_response(c.id, "ok", d1))
        session = agent.finalize(session, assessed_dimensions=["D1"])
        return agent, prov, session

    def test_finalize_records_llm_metadata(self, config, engine):
        agent, _, session = self._scored(config, engine, [GOOD_INTAKE] + [GOOD_CRITERION] * 5)
        assert session.llm_provider == "mock"
        assert session.llm_model == "scripted-1"
        from aipaf import __version__
        assert session.aipaf_version == __version__
        assert session.gate == ApprovalGate.PARTIAL
        assert session.assessed_dimensions == ["D1"]

    def test_executive_summary_mentions_partial(self, config, engine):
        agent, prov, session = self._scored(
            config, engine, [GOOD_INTAKE] + [GOOD_CRITERION] * 5 + ["Sommario narrativo."])
        text = agent.generate_executive_summary(session)
        assert text == "Sommario narrativo."
        prompt = prov.calls[-1]["messages"][-1].content
        assert "PARZIALE" in prompt
        assert "NON VALUTATA" in prompt
        assert len(agent._history) == 2   # le narrative mantengono lo storico

    def test_remediation_dimensions_separated(self, config, engine):
        """Le dimensioni deboli devono essere separate da newline (bug v0.1: ''.join)."""
        low = json.dumps({"score": 1, "confidence": "H", "evidence": "gap", "reasoning": ""})
        agent, prov, session = self._scored(config, engine, [GOOD_INTAKE] + [low] * 5 + ["Piano."])
        agent.generate_remediation_plan(session)
        prompt = prov.calls[-1]["messages"][-1].content
        assert "D1 — Strategic" in prompt
        assert "[D1.C1] score=1.0" in prompt

    def test_remediation_skips_when_all_good(self, config, engine):
        agent, _, session = self._scored(
            config, engine, [GOOD_INTAKE] +
            [json.dumps({"score": 3, "confidence": "H", "evidence": "doc", "reasoning": ""})] * 5)
        assert "Nessun piano" in agent.generate_remediation_plan(session)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
