from pydantic import BaseModel, Field


class FactorDefinition(BaseModel):
    key: str
    label: str
    description: str
    default_weight: float = Field(ge=0, le=1)
    half_life_hours: float | None = None
    evidence_queries: list[str] = Field(default_factory=list)


class DomainPack(BaseModel):
    key: str
    label: str
    description: str
    outcome_types: list[str]
    factors: list[FactorDefinition]
    default_sources: list[str]
    known_failure_modes: list[str]

    def factor_weight_map(self) -> dict[str, float]:
        total = sum(f.default_weight for f in self.factors) or 1
        return {f.key: f.default_weight / total for f in self.factors}

