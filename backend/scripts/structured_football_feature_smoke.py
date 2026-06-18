from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.data_sources.football_match_context import FootballMatchContextSource
from app.data_sources.football_odds_snapshot_csv import FootballOddsSnapshotCsvSource
from app.domain_packs.football import FOOTBALL_PACK
from app.models.football_horizon_profiles import apply_football_horizon_profile
from app.models.weighted_model import WeightedPredictionModel
from app.schemas import PredictionRequest


async def main() -> None:
    request = PredictionRequest(
        question="Predict Norway vs Iraq at the World Cup, who wins?",
        domain="football",
        outcome_type="three_way",
        outcomes=["Norway win", "draw", "Iraq win"],
        event_time=datetime.fromisoformat("2026-06-20T12:00:00"),
        prediction_deadline=datetime.fromisoformat("2026-06-19T12:00:00"),
        context={
            "competitors": ["Norway", "Iraq"],
            "horizon_profile": "T-24h",
            "odds_snapshots": [
                {
                    "match_id": "norway-iraq-test",
                    "home_team": "Norway",
                    "away_team": "Iraq",
                    "snapshot_at": "2026-06-19T11:30:00",
                    "available_at": "2026-06-19T11:35:00",
                    "bookmaker": "public_test_average",
                    "home_odds": "1.75",
                    "draw_odds": "3.60",
                    "away_odds": "5.20",
                    "book_count": "6",
                },
                {
                    "match_id": "norway-iraq-test",
                    "home_team": "Norway",
                    "away_team": "Iraq",
                    "snapshot_at": "2026-06-19T12:30:00",
                    "available_at": "2026-06-19T12:35:00",
                    "bookmaker": "public_test_average",
                    "home_odds": "1.35",
                    "draw_odds": "4.60",
                    "away_odds": "8.80",
                    "book_count": "6",
                }
            ],
            "availability_context": [
                {
                    "team": "Iraq",
                    "player": "Key centre back",
                    "status": "suspended",
                    "expected_minutes_share": 0.85,
                    "player_importance": 0.72,
                    "position_scarcity": 0.8,
                    "replacement_quality": 0.45,
                    "certainty": 0.95,
                }
            ],
            "lineup_context": [
                {
                    "team": "Norway",
                    "lineup_type": "predicted",
                    "xi_strength": 0.78,
                    "continuity_score": 0.72,
                    "surprise_score": 0.05,
                    "certainty": 0.7,
                },
                {
                    "team": "Iraq",
                    "lineup_type": "predicted",
                    "xi_strength": 0.54,
                    "continuity_score": 0.58,
                    "surprise_score": 0.12,
                    "certainty": 0.65,
                },
            ],
        },
    )
    profile = apply_football_horizon_profile(
        request.context,
        request.event_time,
        request.prediction_deadline,
    )
    evidence = []
    for source in [FootballOddsSnapshotCsvSource(), FootballMatchContextSource()]:
        evidence.extend(await source.collect(request, []))

    model = WeightedPredictionModel()
    factors = model.score_factors(FOOTBALL_PACK, request, evidence)
    outcomes = model.predict(request, factors)

    assert profile == "T-24h"
    assert outcomes["Norway win"] > outcomes["Iraq win"]
    assert any(f.key == "market_odds" and f.value > 0.1 and f.weight >= 0.39 for f in factors)
    assert any(f.key == "lineup_availability" and f.value > 0.05 for f in factors)
    market_evidence = next(item for item in evidence if item.impact_area == "market_odds")
    assert "1.75" in market_evidence.claim
    assert "1.35" not in market_evidence.claim
    assert market_evidence.structured_features[0].snapshot_at == datetime.fromisoformat(
        "2026-06-19T11:30:00"
    )
    assert market_evidence.structured_features[0].available_at == datetime.fromisoformat(
        "2026-06-19T11:35:00"
    )
    print(
        {
            "profile": profile,
            "outcomes": outcomes,
            "active_factors": [
                {
                    "key": factor.key,
                    "value": round(factor.value, 4),
                    "weight": round(factor.weight, 4),
                    "confidence": round(factor.confidence, 4),
                }
                for factor in factors
                if factor.evidence_count or abs(factor.value) > 0
            ],
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
