"""
AIPAF — llm/base.py — Interfaccia astratta per i provider LLM
Autore: Lorenzo Lombardi
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class LLMRole(str, Enum):
    USER      = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role:    LLMRole
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMResponse:
    content:       str
    model:         str
    provider:      str
    input_tokens:  Optional[int] = None
    output_tokens: Optional[int] = None
    raw:           Optional[dict] = field(default=None, repr=False)

    @property
    def total_tokens(self) -> Optional[int]:
        if self.input_tokens is not None and self.output_tokens is not None:
            return self.input_tokens + self.output_tokens
        return None


class LLMProvider(ABC):
    #: Ultimo errore incontrato da health_check() (None se l'ultimo check e' andato a buon fine).
    #: La CLI lo mostra: "non raggiungibile" senza il motivo non e' diagnosticabile.
    last_error: Optional[str] = None

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def complete(
        self,
        messages:    list[Message],
        system:      Optional[str] = None,
        *,
        max_tokens:  int   = 2048,
        temperature: float = 0.2,
    ) -> LLMResponse: ...

    def health_check(self) -> bool:
        try:
            resp = self.complete(messages=[Message(LLMRole.USER, "ping")], max_tokens=5)
            self.last_error = None if resp.content else "risposta vuota"
            return bool(resp.content)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(provider={self.provider_name!r}, model={self.model_name!r})"
