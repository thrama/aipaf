"""
AIPAF — Test suite per llm/ layer (base, factory, providers mock)
Eseguire: python -m pytest tests/test_llm.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import MagicMock, patch

import pytest

from aipaf.llm.base import LLMProvider, LLMResponse, LLMRole, Message
from aipaf.llm.factory import available_providers, get_provider

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockProvider(LLMProvider):
    """Provider fittizio per test — restituisce sempre la stessa risposta."""

    def __init__(self, response_text: str = '{"score": 2, "confidence": "M", "evidence": "test", "reasoning": "ok"}'):
        self._response = response_text

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model"

    def complete(self, messages, system=None, *, max_tokens=2048, temperature=0.2) -> LLMResponse:
        return LLMResponse(
            content=self._response,
            model=self.model_name,
            provider=self.provider_name,
            input_tokens=10,
            output_tokens=20,
        )


# ---------------------------------------------------------------------------
# Message e LLMResponse
# ---------------------------------------------------------------------------


class TestMessage:
    def test_to_dict_user(self):
        m = Message(LLMRole.USER, "ciao")
        assert m.to_dict() == {"role": "user", "content": "ciao"}

    def test_to_dict_assistant(self):
        m = Message(LLMRole.ASSISTANT, "risposta")
        assert m.to_dict() == {"role": "assistant", "content": "risposta"}


class TestLLMResponse:
    def test_total_tokens(self):
        r = LLMResponse(content="x", model="m", provider="p", input_tokens=100, output_tokens=50)
        assert r.total_tokens == 150

    def test_total_tokens_none_if_missing(self):
        r = LLMResponse(content="x", model="m", provider="p")
        assert r.total_tokens is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_available_providers(self):
        providers = available_providers()
        assert "claude" in providers
        assert "ollama" in providers

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="non riconosciuto"):
            get_provider("unknown_provider")

    def test_get_ollama_provider(self):
        from aipaf.llm.ollama_provider import OllamaProvider

        p = get_provider("ollama")
        assert isinstance(p, OllamaProvider)
        assert p.provider_name == "ollama"

    def test_get_ollama_with_model_override(self):
        p = get_provider("ollama", model="llama3.1:70b")
        assert p.model_name == "llama3.1:70b"

    def test_get_claude_raises_without_key(self):
        with patch.dict("os.environ", {}, clear=True):
            # Rimuove ANTHROPIC_API_KEY se presente
            import os

            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises((ValueError, ImportError)):
                get_provider("claude")


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------


class TestMockProvider:
    def test_complete_returns_response(self):
        p = MockProvider()
        r = p.complete([Message(LLMRole.USER, "test")])
        assert isinstance(r, LLMResponse)
        assert r.provider == "mock"

    def test_health_check_ok(self):
        p = MockProvider(response_text="pong")
        assert p.health_check() is True

    def test_health_check_fails_on_exception(self):
        p = MockProvider()
        p.complete = MagicMock(side_effect=ConnectionError("offline"))
        assert p.health_check() is False
        assert p.last_error == "ConnectionError: offline"  # il motivo deve restare leggibile

    def test_health_check_ok_clears_error(self):
        p = MockProvider(response_text="pong")
        p.last_error = "vecchio errore"
        assert p.health_check() is True
        assert p.last_error is None


# ---------------------------------------------------------------------------
# _parse_json (utility di agent.py)
# ---------------------------------------------------------------------------


class TestParseJson:
    def _parse(self, text: str) -> dict:
        from aipaf.agent import _parse_json

        return _parse_json(text)

    def test_plain_json(self):
        result = self._parse('{"score": 2, "confidence": "H"}')
        assert result["score"] == 2
        assert result["confidence"] == "H"

    def test_json_with_markdown_fences(self):
        result = self._parse('```json\n{"score": 1}\n```')
        assert result["score"] == 1

    def test_json_with_preamble(self):
        result = self._parse('Ecco la risposta:\n{"score": 3, "confidence": "M"}')
        assert result["score"] == 3

    def test_no_json_raises(self):
        """Nessun fallback silenzioso: un dict vuoto diventerebbe uno score reale."""
        from aipaf.agent import LLMOutputError

        with pytest.raises(LLMOutputError):
            self._parse("nessun JSON qui")

    def test_malformed_json_raises(self):
        from aipaf.agent import LLMOutputError

        with pytest.raises(LLMOutputError):
            self._parse('{"score": 2, "confidence": }')

    def test_json_array_raises(self):
        from aipaf.agent import LLMOutputError

        with pytest.raises(LLMOutputError):
            self._parse("[1, 2, 3]")

    def test_nested_json(self):
        result = self._parse('{"score": 2, "meta": {"source": "test"}}')
        assert result["meta"]["source"] == "test"


# ---------------------------------------------------------------------------
# OllamaProvider (unit, senza server reale)
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    def test_init_default_model(self):
        from aipaf.llm.ollama_provider import OllamaProvider

        p = OllamaProvider()
        assert p.model_name == "llama3.1:8b"
        assert p.provider_name == "ollama"

    def test_init_custom_model(self):
        from aipaf.llm.ollama_provider import OllamaProvider

        p = OllamaProvider(model="qwen3.5:4b")
        assert p.model_name == "qwen3.5:4b"

    def test_complete_connection_error(self):
        from aipaf.llm.ollama_provider import OllamaProvider

        p = OllamaProvider(base_url="http://localhost:19999")
        with pytest.raises(ConnectionError):
            p.complete([Message(LLMRole.USER, "test")])

    def test_health_check_false_when_offline(self):
        from aipaf.llm.ollama_provider import OllamaProvider

        p = OllamaProvider(base_url="http://localhost:19999")
        assert p.health_check() is False
        assert p.model_available() is None

    def test_complete_timeout_error(self):
        """ReadTimeout non e' ConnectionError: deve emergere come TimeoutError esplicito."""
        import requests

        from aipaf.llm.ollama_provider import OllamaProvider

        p = OllamaProvider(base_url="http://localhost:19999", timeout=1)
        with patch("aipaf.llm.ollama_provider.requests.post", side_effect=requests.exceptions.ReadTimeout("slow")):
            with pytest.raises(TimeoutError, match="timeout|non ha risposto"):
                p.complete([Message(LLMRole.USER, "test")])

    def test_model_available_false_when_missing(self):
        from aipaf.llm.ollama_provider import OllamaProvider

        p = OllamaProvider(model="qwen3.5:4b")
        with patch.object(p, "list_local_models", return_value=["llama3.1:8b"]):
            assert p.model_available() is False
            assert p.health_check() is False
        with patch.object(p, "list_local_models", return_value=["qwen3.5:4b", "llama3.1:8b"]):
            assert p.model_available() is True


# ---------------------------------------------------------------------------
# ClaudeProvider (unit, senza API reale)
# ---------------------------------------------------------------------------


class TestClaudeProvider:
    def _make(self, **env):
        from aipaf.llm import claude_provider as cp

        if not cp._ANTHROPIC_AVAILABLE:
            pytest.skip("pacchetto anthropic non installato")
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", **env}, clear=False):
            for k in ("AIPAF_CLAUDE_MODEL", "AIPAF_CLAUDE_TEMPERATURE"):
                if k not in env:
                    import os

                    os.environ.pop(k, None)
            return cp.ClaudeProvider()

    def test_default_model(self):
        from aipaf.llm import claude_provider as cp

        p = self._make()
        assert p.model_name == cp._DEFAULT_MODEL
        assert p.temperature_enabled is True

    def test_model_from_env(self):
        p = self._make(AIPAF_CLAUDE_MODEL="claude-test-model")
        assert p.model_name == "claude-test-model"

    def test_temperature_off_from_env(self):
        p = self._make(AIPAF_CLAUDE_TEMPERATURE="off")
        assert p.temperature_enabled is False

    def test_temperature_auto_fallback(self):
        """Se l'API rifiuta 'temperature', il provider riprova senza e la disabilita."""
        p = self._make()
        import anthropic

        fake = MagicMock()
        fake.content = [MagicMock(text="ok")]
        fake.model = p.model_name
        fake.usage.input_tokens = 1
        fake.usage.output_tokens = 1
        fake.model_dump.return_value = {}

        err = anthropic.BadRequestError(
            message="temperature is not supported", response=MagicMock(status_code=400), body=None
        )
        p._client.messages.create = MagicMock(side_effect=[err, fake])
        r = p.complete([Message(LLMRole.USER, "ping")])
        assert r.content == "ok"
        assert p.temperature_enabled is False
        first_call, second_call = p._client.messages.create.call_args_list
        assert "temperature" in first_call.kwargs
        assert "temperature" not in second_call.kwargs

    def test_temperature_sdk_typeerror_fallback(self):
        """SDK anthropic >= 1.0: messages.create() non ha piu' 'temperature' → TypeError."""
        p = self._make()
        fake = MagicMock()
        fake.content = [MagicMock(text="ok")]
        fake.model = p.model_name
        fake.usage.input_tokens = 1
        fake.usage.output_tokens = 1
        fake.model_dump.return_value = {}
        err = TypeError("Messages.create() got an unexpected keyword argument 'temperature'")
        p._client.messages.create = MagicMock(side_effect=[err, fake])
        r = p.complete([Message(LLMRole.USER, "ping")])
        assert r.content == "ok"
        assert p.temperature_enabled is False
        assert "temperature" not in p._client.messages.create.call_args_list[1].kwargs

    def test_unrelated_typeerror_propagates(self):
        p = self._make()
        p._client.messages.create = MagicMock(side_effect=TypeError("qualcos'altro"))
        with pytest.raises(TypeError, match="qualcos'altro"):
            p.complete([Message(LLMRole.USER, "ping")])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
