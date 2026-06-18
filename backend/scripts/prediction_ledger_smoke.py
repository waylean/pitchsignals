from __future__ import annotations

from datetime import datetime
from pathlib import Path
import asyncio
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.evidence.store import InMemoryEvidenceStore  # noqa: E402
from app.feedback.analyzer import FeedbackAnalyzer  # noqa: E402
from app.feedback.ledger import PredictionLedger  # noqa: E402
from app.schemas import (  # noqa: E402
    EvidenceItem,
    FactorScore,
    FeedbackRequest,
    OutcomeType,
    PredictionRequest,
    PredictionResponse,
)


async def main() -> None:
    path = ROOT / "work" / "prediction_ledger_smoke.jsonl"
    if path.exists():
        path.unlink()
    ledger = PredictionLedger(path)
    request = PredictionRequest(
        question="Norway vs Iraq, who wins?",
        domain="football",
        outcome_type=OutcomeType.THREE_WAY,
        event_time=datetime.fromisoformat("2026-06-20T20:00:00"),
        prediction_deadline=datetime.fromisoformat("2026-06-19T20:00:00"),
        context={
            "horizon_profile": "T-24h",
            "competitors": ["Norway", "Iraq"],
        },
    )
    response = PredictionResponse(
        task_id="ledger-smoke-001",
        domain="football",
        normalized_question="Norway vs Iraq, who wins?",
        outcomes={"home_win": 0.62, "draw": 0.23, "away_win": 0.15},
        model_status="evidence_directional",
        confidence=0.71,
        data_coverage=0.82,
        freshness=0.77,
        model_agreement=0.69,
        factors=[
            FactorScore(
                key="market_odds",
                label="Market Odds",
                value=0.45,
                weight=0.25,
                confidence=0.9,
                evidence_count=3,
                rationale="Synthetic smoke signal.",
            )
        ],
        evidence=[
            EvidenceItem(
                evidence_id="evidence-001",
                claim="Synthetic odds snapshot was available before deadline.",
                source="smoke",
                source_url="https://example.test/smoke",
                impact_area="market_odds",
                source_reliability=0.8,
                recency_score=0.9,
                confidence=0.85,
            )
        ],
        uncertainties=[],
        workflow_trace=["task_intake", "domain_pack:football", "probability_estimation"],
        distribution_metrics={
            "ensemble_profile": {
                "profile_id": "football_research_alpha",
                "version": "0.6.0",
            }
        },
    )
    prediction_event = ledger.record_prediction(request, response)
    feedback = FeedbackRequest(
        task_id=response.task_id,
        actual_outcome="home_win",
        resolved_at=datetime.fromisoformat("2026-06-20T22:00:00"),
        post_event_notes="Synthetic final score confirmed.",
    )
    feedback_event = ledger.record_feedback(
        feedback,
        {
            "task_id": response.task_id,
            "actual_outcome": feedback.actual_outcome,
            "metrics": {
                "brier_score": 0.216,
                "log_loss": 0.478036,
                "predicted_probability": 0.62,
                "top_prediction": "home_win",
                "was_top_prediction_correct": True,
            },
        },
    )
    events = ledger.events(task_id=response.task_id)
    summary = ledger.task_summary(response.task_id)
    assert prediction_event.event_type == "prediction_created"
    assert feedback_event.event_type == "outcome_resolved"
    assert len(events) == 2
    assert summary["has_prediction"] is True
    assert summary["has_resolution"] is True
    assert summary["latest_prediction"]["response"]["pick"] == "home_win"
    assert summary["latest_resolution"]["feedback"]["actual_outcome"] == "home_win"
    analyzer = FeedbackAnalyzer(InMemoryEvidenceStore(), ledger)
    fallback = await analyzer.analyze(
        FeedbackRequest(
            task_id=response.task_id,
            actual_outcome="home_win",
            resolved_at=datetime.fromisoformat("2026-06-20T22:30:00"),
        )
    )
    assert fallback["prediction_source"] == "persistent_ledger"
    assert fallback["metrics"]["top_prediction"] == "home_win"
    print(
        {
            "ledger": str(path),
            "event_count": len(ledger.events(task_id=response.task_id)),
            "prediction_event": prediction_event.event_id,
            "feedback_event": feedback_event.event_id,
            "fallback_source": fallback["prediction_source"],
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
