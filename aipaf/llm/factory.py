"""
AIPAF — AI Project Assessment Framework
llm/factory.py — Factory per la selezione del provider LLM

Uso:
    provider = get_provider("claude")
    provider = get_provider("ollama", model="llama3.1:70b")
    provider = get_provider("ollama", base_url="http://gpu-server:11434")

Autore: Lorenzo Lombardi
"""

from __future__ import annotations

from typing import Optional

from .base import LLMProvider
from .claude_provider import ClaudeProvider
from .ollama_provider import OllamaProvider

_REGISTRY: dict[str, type[LLMProvider]] = {
    "claude": ClaudeProvider,
    "ollama": OllamaProvider,
}


def get_provider(
    name:  str,
    model: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """
    Istanzia il provider LLM richiesto.

    Args:
        name:     'claude' | 'ollama'
        model:    Override del modello di default
        **kwargs: Parametri aggiuntivi per il costruttore
                  (es. base_url per Ollama, api_key per Claude)
    """
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Provider '{name}' non riconosciuto. "
            f"Disponibili: {', '.join(_REGISTRY)}"
        )
    if model:
        kwargs["model"] = model
    return cls(**kwargs)


def available_providers() -> list[str]:
    return list(_REGISTRY.keys())
