from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.core.entities import infer_competitors
from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest, StructuredFootballFeature


class FootballScheduleContextSource:
    """Convert selected fixture odds from the schedule UI into evidence."""

    name = "football_schedule_context"

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if request.domain != "football":
            return []
        odds = request.context.get("schedule_odds")
        if not isinstance(odds, dict) or not odds.get("available"):
            return []
        first, second = infer_competitors(request)
        if not first or not second:
            return []
        try:
            first_odds = float(odds["home_odds"])
            draw_odds = float(odds["draw_odds"])
            second_odds = float(odds["away_odds"])
        except (KeyError, TypeError, ValueError):
            return []
        if min(first_odds, draw_odds, second_odds) <= 1:
            return []

        raw_first = 1 / first_odds
        raw_draw = 1 / draw_odds
        raw_second = 1 / second_odds
        total = raw_first + raw_draw + raw_second
        first_prob = raw_first / total
        draw_prob = raw_draw / total
        second_prob = raw_second / total
        edge = (first_prob - second_prob) / max(first_prob + second_prob, 0.001)
        magnitude = min(abs(edge), 1.0)
        now = datetime.now(timezone.utc)
        match_id = str(request.context.get("match_id") or "")
        bookmaker_count = int(odds.get("bookmaker_count") or 0)
        feature_value: dict[str, Any] = {
            "home_team": first,
            "away_team": second,
            "first_team": first,
            "second_team": second,
            "home_odds": first_odds,
            "draw_odds": draw_odds,
            "away_odds": second_odds,
            "first_odds": first_odds,
            "second_odds": second_odds,
            "home_prob": first_prob,
            "draw_prob": draw_prob,
            "away_prob": second_prob,
            "first_prob": first_prob,
            "second_prob": second_prob,
            "overround": total - 1,
            "direction": 1.0 if edge >= 0 else -1.0,
            "magnitude": magnitude,
            "book_count": float(bookmaker_count),
            "odds_type": "schedule_h2h",
            "bookmaker": "schedule_average_consensus",
            "extreme_favorite": min(first_odds, second_odds) <= 1.35 or max(first_odds, second_odds) >= 8.0,
        }
        feature = StructuredFootballFeature(
            feature_type="odds",
            impact_area="market_odds",
            feature_value=feature_value,
            direction=1.0 if edge >= 0 else -1.0,
            magnitude=magnitude,
            confidence=0.92,
            feature_confidence=0.92,
            match_id=match_id,
            snapshot_at=now,
            available_at=now,
            prediction_deadline=request.prediction_deadline,
            leakage_risk="medium",
            extraction_method="schedule_selected_h2h",
            source_name=self.name,
            source_provenance={
                "match_id": match_id,
                "sport_key": request.context.get("the_odds_api_sport_key"),
                "bookmaker_count": bookmaker_count,
            },
            rationale=(
                f"Selected fixture market gives {first} {first_prob:.1%}, draw {draw_prob:.1%}, "
                f"{second} {second_prob:.1%} after overround adjustment."
            ),
        )
        claim = (
            f"Selected fixture 1X2 odds: {first} {first_odds:.2f}, draw {draw_odds:.2f}, "
            f"{second} {second_odds:.2f}; probabilities first={first_prob:.3f}, "
            f"draw={draw_prob:.3f}, second={second_prob:.3f}; bookmakers={bookmaker_count}."
        )
        return [
            EvidenceItem(
                claim=claim,
                source=self.name,
                source_query="selected schedule odds",
                evidence_stage="verified_candidate",
                raw_excerpt=claim,
                verifier_notes=["Parsed from the selected schedule card's h2h odds snapshot."],
                published_at=now,
                impact_area="market_odds",
                source_reliability=0.9,
                recency_score=0.96,
                corroboration_count=bookmaker_count,
                confidence=0.92,
                structured_features=[feature],
            )
        ]
