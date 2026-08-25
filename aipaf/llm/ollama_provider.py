"""
AIPAF — AI Project Assessment Framework
llm/ollama_provider.py — Provider per modelli locali via Ollama

Richiede: pip install requests
          Ollama in esecuzione su localhost:11434 (default)

Autore: Lorenzo Lombardi
"""

from __future__ import annotations

import os
from typing import Optional

from .base import LLMProvider, LLMResponse, Message

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# Modelli consigliati per task AIPAF (in ordine di qualita' vs peso):
#
#   qwen3.5:9b     — buon compromesso su 16 GB RAM
#   qwen3.5:4b     — sviluppo e test, notebook 8-16 GB
#   llama3.1:8b    — alternativa generalista
#
# Nota: per reasoning normativo complesso (EU AI Act, ISO 42001)
# la qualita' e' inferiore rispetto a Claude. Provider e modello usati
# vengono registrati nella sessione e nel report.

_DEFAULT_MODEL    = "llama3.1:8b"
_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """
    Provider per modelli locali via Ollama REST API (/api/chat).

    Vantaggi: i dati non escono dall'infrastruttura aziendale, nessun costo per token.
    Limiti:   qualita' reasoning normativo inferiore a Claude; hardware dedicato >13B.
    """

    def __init__(
        self,
        model:    str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
        timeout:  int = 120,
    ):
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "Il pacchetto 'requests' non e' installato. "
                "Eseguire: pip install requests"
            )
        self._model    = model
        self._base_url = (base_url or os.environ.get("OLLAMA_HOST") or _DEFAULT_BASE_URL).rstrip("/")
        self._timeout  = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    def complete(
        self,
        messages:    list[Message],
        system:      Optional[str] = None,
        *,
        max_tokens:  int   = 2048,
        temperature: float = 0.2,
    ) -> LLMResponse:
        ollama_messages: list[dict] = []
        if system:
            ollama_messages.append({"role": "system", "content": system})
        ollama_messages.extend(m.to_dict() for m in messages)

        payload = {
            "model":   self._model,
            "messages": ollama_messages,
            "stream":  False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            # ReadTimeout NON e' sottoclasse di ConnectionError: va gestito a parte
            raise TimeoutError(
                f"Ollama non ha risposto entro {self._timeout}s su {self._base_url} "
                f"(modello {self._model}). Aumentare timeout o usare un modello piu' leggero."
            )
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Impossibile connettersi a Ollama su {self._base_url}. "
                "Verificare che Ollama sia in esecuzione: ollama serve"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Errore Ollama API: {e}")

        data = response.json()
        content = data.get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            input_tokens=data.get("prompt_eval_count"),
            output_tokens=data.get("eval_count"),
            raw=data,
        )

    # ------------------------------------------------------------------
    # Diagnostica (nessun output diretto: la presentazione spetta alla CLI)
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """True se Ollama risponde E il modello configurato e' disponibile."""
        state = self.model_available()
        if state is None:
            self.last_error = f"Ollama non raggiungibile su {self._base_url}"
        elif state is False:
            self.last_error = f"modello '{self._model}' non presente (ollama pull {self._model})"
        else:
            self.last_error = None
        return state is True

    def model_available(self) -> Optional[bool]:
        """
        True  → server raggiungibile e modello presente
        False → server raggiungibile ma modello assente
        None  → server non raggiungibile
        """
        models = self.list_local_models()
        if models is None:
            return None
        model_base = self._model.split(":")[0]
        return any(m == self._model or m.startswith(model_base + ":") or m == model_base
                   for m in models)

    def list_local_models(self) -> Optional[list[str]]:
        """Modelli disponibili localmente; None se Ollama non risponde."""
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return None
