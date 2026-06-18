from datetime import datetime, timezone

from app.schemas import EvidenceItem


def score_recency(published_at: datetime | None, half_life_hours: float | None) -> float:
    if not published_at or not half_life_hours:
        return 0.5
    now = datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_hours = max((now - published_at).total_seconds() / 3600, 0)
    return max(0.05, 0.5 ** (age_hours / half_life_hours))


def evidence_quality(item: EvidenceItem) -> float:
    corroboration = min(item.corroboration_count / 3, 1)
    contradiction_penalty = min(item.contradiction_count * 0.2, 0.6)
    score = (
        item.source_reliability * 0.35
        + item.recency_score * 0.25
        + item.confidence * 0.25
        + corroboration * 0.15
        - contradiction_penalty
    )
    return max(0, min(score, 1))

