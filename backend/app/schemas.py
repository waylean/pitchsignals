from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class OutcomeType(StrEnum):
    BINARY = "binary"
    THREE_WAY = "three_way"
    MULTI_CLASS = "multi_class"
    NUMERIC = "numeric"
    SCORELINE = "scoreline"


class PredictionRequest(BaseModel):
    question: str
    domain: str = Field(default="football")
    outcome_type: OutcomeType = Field(default=OutcomeType.THREE_WAY)
    outcomes: list[str] = Field(default_factory=lambda: ["home_win", "draw", "away_win"])
    event_time: datetime | None = None
    prediction_deadline: datetime | None = None
    context: dict[str, Any] = Field(default_factory=dict)


class FootballScheduleMatch(BaseModel):
    match_id: str
    sport_key: str
    league: str | None = None
    commence_time: datetime
    home_team: str
    away_team: str
    neutral_site: bool = False
    odds: dict[str, Any] = Field(default_factory=dict)
    source: str = "the_odds_api"


class FootballScheduleResponse(BaseModel):
    status: str = "ready"
    date: str
    timezone: str
    matches: list[FootballScheduleMatch] = Field(default_factory=list)
    source: str = "the_odds_api"
    notes: list[str] = Field(default_factory=list)


class StructuredFootballFeature(BaseModel):
    feature_type: str
    impact_area: str
    feature_value: dict[str, Any] = Field(default_factory=dict)
    direction: float = Field(default=0.0, ge=-1, le=1)
    magnitude: float = Field(default=0.0, ge=0, le=1)
    confidence: float = Field(default=0.0, ge=0, le=1)
    feature_confidence: float | None = Field(default=None, ge=0, le=1)
    team: str | None = None
    player: str | None = None
    referee: str | None = None
    match_id: str | None = None
    snapshot_at: datetime | None = None
    available_at: datetime | None = None
    prediction_deadline: datetime | None = None
    leakage_risk: str = "medium"
    extraction_method: str = "structured_context"
    source_name: str | None = None
    source_url: str | None = None
    license_note: str | None = None
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    horizon_allowed: bool = True
    rationale: str | None = None


class EvidenceItem(BaseModel):
    evidence_id: str | None = None
    claim: str
    source: str
    source_url: str | None = None
    license_note: str | None = None
    source_provenance: dict[str, Any] = Field(default_factory=dict)
    source_query: str | None = None
    evidence_stage: str = Field(default="candidate")
    dedupe_key: str | None = None
    raw_excerpt: str | None = None
    verifier_notes: list[str] = Field(default_factory=list)
    published_at: datetime | None = None
    collected_at: datetime = Field(default_factory=datetime.utcnow)
    deadline_compliant: bool = True
    impact_area: str
    source_reliability: float = Field(ge=0, le=1)
    recency_score: float = Field(ge=0, le=1)
    corroboration_count: int = Field(default=0, ge=0)
    contradiction_count: int = Field(default=0, ge=0)
    confidence: float = Field(ge=0, le=1)
    structured_features: list[StructuredFootballFeature] = Field(default_factory=list)


class FactorScore(BaseModel):
    key: str
    label: str
    value: float = Field(ge=-1, le=1)
    weight: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    evidence_count: int = Field(ge=0)
    rationale: str


class PredictionResponse(BaseModel):
    task_id: str
    domain: str
    normalized_question: str
    outcomes: dict[str, float]
    pick: str | None = None
    pick_probability: float | None = Field(default=None, ge=0, le=1)
    model_status: str = "baseline"
    confidence: float = Field(ge=0, le=1)
    data_coverage: float = Field(ge=0, le=1)
    data_completeness: float = Field(default=0, ge=0, le=1)
    freshness: float = Field(ge=0, le=1)
    model_agreement: float = Field(ge=0, le=1)
    horizon_profile: str | None = None
    evidence_gate_status: str = "unknown"
    missing_evidence: list[str] = Field(default_factory=list)
    factors: list[FactorScore]
    evidence: list[EvidenceItem]
    uncertainties: list[str]
    workflow_trace: list[str]
    research_review: dict[str, Any] | None = None
    model_runs: list[dict[str, Any]] = Field(default_factory=list)
    distribution_metrics: dict[str, Any] = Field(default_factory=dict)


class FeedbackRequest(BaseModel):
    task_id: str
    actual_outcome: str
    resolved_at: datetime = Field(default_factory=datetime.utcnow)
    post_event_notes: str | None = None


class FeedbackMetrics(BaseModel):
    brier_score: float | None = None
    log_loss: float | None = None
    predicted_probability: float | None = None
    top_prediction: str | None = None
    was_top_prediction_correct: bool | None = None
