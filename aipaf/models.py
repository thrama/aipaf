"""
AIPAF — AI Project Assessment Framework
models.py — Pydantic models

Contiene anche le soglie di scoring/gate: sono l'unica fonte di verità,
importate da engine.py e report.py. Non duplicarle altrove.

Autore: Lorenzo Lombardi
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Soglie (single source of truth)
# ---------------------------------------------------------------------------

DIMENSION_CRITICAL_THRESHOLD = 0.5   # dimensione < soglia → RED forzato
GATE_YELLOW_THRESHOLD        = 1.0   # composito < soglia → RED
GATE_GREEN_THRESHOLD         = 2.0   # composito < soglia → YELLOW, altrimenti GREEN
REMEDIATION_THRESHOLD        = 2.0   # dimensione < soglia → entra nel remediation plan

ALL_DIMENSION_IDS = ("D1", "D2", "D3", "D4", "D5", "D6")


# ---------------------------------------------------------------------------
# Enumerazioni
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    HIGH   = "H"
    MEDIUM = "M"
    LOW    = "L"

class ApprovalGate(str, Enum):
    GREEN   = "GREEN"
    YELLOW  = "YELLOW"
    RED     = "RED"
    PARTIAL = "PARTIAL"   # assessment incompleto: nessun gate definitivo

class EUAIActLevel(str, Enum):
    UNACCEPTABLE = "UNACCEPTABLE"
    HIGH_RISK    = "HIGH_RISK"
    LIMITED_RISK = "LIMITED_RISK"
    MINIMAL_RISK = "MINIMAL_RISK"

class Sector(str, Enum):
    DEFAULT         = "default"
    BANKING_FINANCE = "banking_finance"
    HEALTHCARE      = "healthcare"
    MANUFACTURING   = "manufacturing"
    PUBLIC_ADMIN    = "public_admin"
    MEDIA_TELECOM   = "media_telecom"


# ---------------------------------------------------------------------------
# Risposte e score
# ---------------------------------------------------------------------------

class CriterionResponse(BaseModel):
    """
    Risposta a un singolo criterio.

    not_applicable=True marca il criterio come N/A: viene escluso dal
    denominatore della dimensione (non conta né a favore né contro).
    Per i criteri N/A score e confidence sono ignorati.
    """
    criterion_id:   str
    score:          int             = Field(default=0, ge=0, le=3)
    confidence:     ConfidenceLevel = ConfidenceLevel.MEDIUM
    evidence:       str             = ""
    notes:          Optional[str]   = None
    not_applicable: bool            = False

    @property
    def effective_score(self) -> float:
        if self.not_applicable:
            return 0.0
        multipliers = {
            ConfidenceLevel.HIGH:   1.0,
            ConfidenceLevel.MEDIUM: 0.8,
            ConfidenceLevel.LOW:    0.0,
        }
        return self.score * multipliers[self.confidence]

    @property
    def data_needed_flag(self) -> bool:
        return (not self.not_applicable) and self.confidence == ConfidenceLevel.LOW


class DimensionScore(BaseModel):
    """
    Score aggregato di una dimensione.

    assessed=False   → la dimensione non è stata valutata in questa sessione
                       (es. --dims parziale). Esclusa da composito, gate e pattern.
    applicable=False → tutti i criteri della dimensione sono N/A.
                       Esclusa da composito e gate, con nota esplicita.
    In entrambi i casi raw_score vale 0.0 solo come placeholder.
    """
    dimension_id:       str
    dimension_name:     str
    raw_score:          float
    weight_pct:         int
    assessed:           bool                    = True
    applicable:         bool                    = True
    criteria_responses: list[CriterionResponse] = Field(default_factory=list)
    flags:              list[str]               = Field(default_factory=list)
    data_needed:        list[str]               = Field(default_factory=list)
    not_applicable:     list[str]               = Field(default_factory=list)

    @property
    def counts(self) -> bool:
        """True se la dimensione concorre a composito e gate."""
        return self.assessed and self.applicable

    @property
    def is_critical(self) -> bool:
        return self.counts and self.raw_score < DIMENSION_CRITICAL_THRESHOLD


class PatternAlert(BaseModel):
    pattern_id:         str
    name:               str
    label:              str
    description:        str
    recommended_action: str
    triggered:          bool = False


class NormativeRedFlag(BaseModel):
    criterion_id:   str
    criterion_name: str
    norm_reference: str
    description:    str


class ProjectIntake(BaseModel):
    project_name:           str
    use_case_description:   str
    project_owner:          str
    sector:                 Sector       = Sector.DEFAULT
    ai_system_type:         str          = ""
    eu_ai_act_level:        EUAIActLevel = EUAIActLevel.MINIMAL_RISK
    project_stage:          str          = ""
    user_count_estimate:    Optional[int] = None
    vulnerable_subjects:    bool         = False
    personal_data_treated:  bool         = False
    sensitive_data_treated: bool         = False
    assessment_date:        str          = ""


class AssessmentSession(BaseModel):
    intake:              Optional[ProjectIntake] = None
    responses:           list[CriterionResponse] = Field(default_factory=list)
    dimension_scores:    list[DimensionScore]    = Field(default_factory=list)
    composite_score:     Optional[float]         = None
    gate:                Optional[ApprovalGate]  = None
    pattern_alerts:      list[PatternAlert]      = Field(default_factory=list)
    normative_red_flags: list[NormativeRedFlag]  = Field(default_factory=list)

    # Copertura dell'assessment
    assessed_dimensions: list[str]              = Field(default_factory=list)
    partial:             bool                   = False
    provisional_gate:    Optional[ApprovalGate] = None   # gate sulle sole dimensioni valutate (se partial)

    # Tracciabilità: quale LLM ha interpretato le risposte
    llm_provider:  Optional[str] = None
    llm_model:     Optional[str] = None
    aipaf_version: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        answered = {r.criterion_id.split(".")[0] for r in self.responses}
        return set(ALL_DIMENSION_IDS).issubset(answered)

    @property
    def unassessed_dimensions(self) -> list[str]:
        return [d.dimension_id for d in self.dimension_scores if not d.assessed]

    @property
    def not_applicable_dimensions(self) -> list[str]:
        return [d.dimension_id for d in self.dimension_scores if d.assessed and not d.applicable]

    def get_response(self, criterion_id: str) -> Optional[CriterionResponse]:
        for r in self.responses:
            if r.criterion_id == criterion_id:
                return r
        return None

    def upsert_response(self, response: CriterionResponse) -> None:
        for i, r in enumerate(self.responses):
            if r.criterion_id == response.criterion_id:
                self.responses[i] = response
                return
        self.responses.append(response)


# ---------------------------------------------------------------------------
# Configurazione framework
# ---------------------------------------------------------------------------

class ScoringGuide(BaseModel):
    s0: str = Field(alias="0")
    s1: str = Field(alias="1")
    s2: str = Field(alias="2")
    s3: str = Field(alias="3")
    model_config = {"populate_by_name": True}


class CriterionConfig(BaseModel):
    id:             str
    name:           str
    description:    str
    weight_pct:     int
    source:         str
    question_id:    str
    scoring_guide:  ScoringGuide
    flags:          list[str]    = Field(default_factory=list)
    notes:          Optional[str] = None
    skip_condition: Optional[str] = None   # es. "personal_data_treated == false"


class DimensionConfig(BaseModel):
    id:               str
    name:             str
    description:      str
    implicit_purpose: str
    compiled_by:      str
    criteria:         list[CriterionConfig]

    @model_validator(mode="after")
    def criteria_weights_sum_100(self) -> "DimensionConfig":
        total = sum(c.weight_pct for c in self.criteria)
        if total != 100:
            raise ValueError(f"{self.id}: i pesi dei criteri devono sommare a 100, somma: {total}")
        return self


class SectorWeights(BaseModel):
    D1: int
    D2: int
    D3: int
    D4: int
    D5: int
    D6: int

    @model_validator(mode="after")
    def weights_sum_100(self) -> "SectorWeights":
        total = self.D1+self.D2+self.D3+self.D4+self.D5+self.D6
        if total != 100:
            raise ValueError(f"I pesi devono sommare a 100, somma: {total}")
        return self

    def as_map(self) -> dict[str, int]:
        return {"D1": self.D1, "D2": self.D2, "D3": self.D3,
                "D4": self.D4, "D5": self.D5, "D6": self.D6}


class FrameworkConfig(BaseModel):
    dimensions:     list[DimensionConfig]
    sector_weights: dict[str, SectorWeights]
    pattern_alerts: dict
    framework_version: str = ""

    @property
    def dimension_map(self) -> dict[str, DimensionConfig]:
        return {d.id: d for d in self.dimensions}

    @property
    def dimension_ids(self) -> list[str]:
        return [d.id for d in self.dimensions]

    @property
    def criterion_ids(self) -> set[str]:
        return {c.id for d in self.dimensions for c in d.criteria}

    def get_criterion(self, criterion_id: str) -> Optional[CriterionConfig]:
        parts = criterion_id.split(".")
        if len(parts) < 2:
            return None
        dim = self.dimension_map.get(parts[0])
        if dim is None:
            return None
        for c in dim.criteria:
            if c.id == criterion_id:
                return c
        return None
