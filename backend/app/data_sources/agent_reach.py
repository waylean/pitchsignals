from datetime import datetime

from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest


class AgentReachSource:
    """Adapter boundary for Agent-Reach powered collection.

    This scaffold keeps the contract explicit. A production implementation can
    call Agent-Reach upstream tools: Exa, Jina Reader, RSS, Twitter, Reddit,
    Bilibili, YouTube, GitHub, and other configured channels.
    """

    name = "agent_reach"

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        return [
            EvidenceItem(
                claim=f"Agent-Reach adapter not configured yet for query: {query.text}",
                source=self.name,
                source_query=query.text,
                evidence_stage="collection_gap",
                published_at=None,
                collected_at=datetime.utcnow(),
                impact_area="collection_gap",
                source_reliability=0.4,
                recency_score=0.5,
                corroboration_count=0,
                contradiction_count=0,
                confidence=0.25,
            )
            for query in queries[:3]
        ]
