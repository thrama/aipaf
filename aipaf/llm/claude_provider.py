"""
AIPAF — AI Project Assessment Framework
llm/claude_provider.py — Provider per Anthropic Claude

Richiede: pip install anthropic

Configurazione:
  ANTHROPIC_API_KEY    — obbligatoria
  AIPAF_CLAUDE_MODEL   — modello (default: claude-sonnet-4-6)
  AIPAF_CLAUDE_TEMPERATURE — "auto" (default) | "off"
      Alcuni modelli rifiutano parametri di sampling non di default, e
      l'SDK anthropic >= 1.0 ha rimosso del tutto il parametro `temperature`
      da messages.create() (TypeError). In modalità "auto" il provider prova
      a passarla e, se SDK o API la rifiutano, la disabilita per il resto
      della sessione. Con SDK >= 1.0 è di fatto sempre disabilitata.

Autore: Lorenzo Lombardi
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import LLMProvider, LLMResponse, Message

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

log = logging.getLogger("aipaf.llm.claude")

_DEFAULT_MODEL   = "claude-sonnet-4-6"
_ENV_MODEL       = "AIPAF_CLAUDE_MODEL"
_ENV_TEMPERATURE = "AIPAF_CLAUDE_TEMPERATURE"


def default_claude_model() -> str:
    return os.environ.get(_ENV_MODEL) or _DEFAULT_MODEL


class ClaudeProvider(LLMProvider):
    """
    Provider per Claude via Anthropic API.

    La API key viene letta da ANTHROPIC_API_KEY (variabile d'ambiente).
    Non passare mai la chiave nel codice.
    """

    def __init__(
        self,
        model:   Optional[str] = None,
        api_key: Optional[str] = None,
        use_temperature: Optional[bool] = None,
    ):
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Il pacchetto 'anthropic' non e' installato. "
                "Eseguire: pip install anthropic"
            )
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "ANTHROPIC_API_KEY non trovata. "
                "Impostare la variabile d'ambiente prima di avviare l'agent."
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._model  = model or default_claude_model()

        if use_temperature is None:
            use_temperature = os.environ.get(_ENV_TEMPERATURE, "auto").lower() != "off"
        self._use_temperature = use_temperature

    @property
    def provider_name(self) -> str:
        return "claude"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def temperature_enabled(self) -> bool:
        return self._use_temperature

    def complete(
        self,
        messages:    list[Message],
        system:      Optional[str] = None,
        *,
        max_tokens:  int   = 2048,
        temperature: float = 0.2,
    ) -> LLMResponse:
        kwargs: dict = {
            "model":      self._model,
            "max_tokens": max_tokens,
            "messages":   [m.to_dict() for m in messages],
        }
        if system:
            kwargs["system"] = system

        if self._use_temperature:
            try:
                response = self._client.messages.create(temperature=temperature, **kwargs)
            except (anthropic.BadRequestError, TypeError) as e:
                # BadRequestError: il modello non accetta parametri di sampling.
                # TypeError:       l'SDK (>= 1.0) non ha piu' il parametro.
                if "temperature" not in str(e).lower():
                    raise
                log.warning("'temperature' non supportata (%s); disabilitata per questa sessione.",
                            type(e).__name__)
                self._use_temperature = False
                response = self._client.messages.create(**kwargs)
        else:
            response = self._client.messages.create(**kwargs)

        content = "".join(getattr(block, "text", "") for block in response.content)

        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            raw=response.model_dump() if hasattr(response, "model_dump") else None,
        )
