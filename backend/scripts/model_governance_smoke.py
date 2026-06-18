from __future__ import annotations

import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.schemas import PredictionRequest
from app.workflows.orchestrator import PredictionWorkflow


async def main() -> None:
    workflow = PredictionWorkflow()
    football = await workflow.predict(
        PredictionRequest(
            question="Predict Norway vs Iraq, who wins?",
            domain="football",
            outcome_type="three_way",
            outcomes=["Norway win", "draw", "Iraq win"],
            context={
                "competitors": ["Norway", "Iraq"],
                "odds_snapshots": [
                    {
                        "home_team": "Norway",
                        "away_team": "Iraq",
                        "as_of": "2026-06-19T12:00:00",
                        "home_odds": "1.75",
                        "draw_odds": "3.60",
                        "away_odds": "5.20",
                    }
                ],
            },
        )
    )
    finance = await workflow.predict(
        PredictionRequest(
            question="Will this asset go up?",
            domain="finance",
            outcome_type="binary",
            outcomes=["yes", "no"],
            context={"expected_goals": {"first": 1.85, "second": 0.85}},
        )
    )
    football_models = {run["model_id"] for run in football.model_runs}
    finance_models = {run["model_id"] for run in finance.model_runs}
    assert "football_market_only" in football_models
    assert "football_elo_strength_baseline" in football_models
    assert "football_poisson_score_grid" in football_models
    assert "football_dixon_coles_score_grid" in football_models
    assert any("market_implied_xg" in run["rationale"] for run in football.model_runs)
    assert "football_poisson_score_grid" not in finance_models
    assert "football_market_only" not in finance_models
    assert football.distribution_metrics["auxiliary_model_count"] >= 4
    assert football.distribution_metrics["ensemble_profile"]["profile_id"] == "football_research_alpha"
    assert football.distribution_metrics["ensemble_profile"]["version"] == "0.6.0"
    assert finance.distribution_metrics["auxiliary_model_count"] == 0
    print(
        {
            "football_outcomes": football.outcomes,
            "football_model_runs": football.model_runs,
            "football_distribution_metrics": football.distribution_metrics,
            "finance_model_runs": finance.model_runs,
            "finance_distribution_metrics": finance.distribution_metrics,
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
