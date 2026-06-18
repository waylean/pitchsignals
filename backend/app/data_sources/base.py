from typing import Protocol

from pydantic import BaseModel

from app.schemas import EvidenceItem, PredictionRequest


class EvidenceQuery(BaseModel):
    text: str
    impact_area: str
    factor_label: str | None = None


class DataSource(Protocol):
    name: str

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        """Collect candidate evidence for a prediction task."""
