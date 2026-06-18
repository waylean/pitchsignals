from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.core.config import settings
from app.core.entities import infer_competitors
from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest, StructuredFootballFeature


class TheOddsApiOddsSource:
    """Optional 1X2 odds adapter for The Odds API.

    This source is disabled unless THE_ODDS_API_KEY is configured. It requests
    the h2h market for a sport key, then converts bookmaker prices into a single
    overround-adjusted directional market feature.
    """

    name = "the_odds_api"
    base_url = "https://api.the-odds-api.com/v4"

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if request.domain != "football":
            return []
        api_key = str(request.context.get("the_odds_api_key") or settings.the_odds_api_key or "").strip()
        if not api_key:
            return [self._gap("The Odds API key is not configured.")]

        first, second = infer_competitors(request)
        if not first or not second:
            return [self._gap("The Odds API odds require two competitors.")]

        sport_key = str(request.context.get("the_odds_api_sport_key") or settings.the_odds_api_default_sport_key)
        regions = str(request.context.get("the_odds_api_regions") or settings.the_odds_api_regions)
        params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        url = f"{self.base_url}/sports/{sport_key}/odds"
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                events = response.json()
        except Exception as exc:
            return [self._gap(f"The Odds API odds collection failed: {exc!r}")]

        event = self._find_event(events, first, second, request.context.get("match_id"))
        if not event:
            return [self._gap(f"No The Odds API h2h odds event matched {first} vs {second}.")]

        commence_time = self._parse_datetime(event.get("commence_time"))
        if (
            commence_time
            and not request.context.get("allow_started_the_odds_api_events")
            and self._after(datetime.now(timezone.utc), commence_time)
        ):
            return [self._gap("Matched The Odds API event has already started; live/current odds were excluded.")]

        parsed = self._parse_event(event, first, second)
        if not parsed:
            return [self._gap("Matched The Odds API event has no usable h2h 1X2 prices.")]

        snapshot_at = datetime.now(timezone.utc)
        first_team = str(parsed["first_team"])
        second_team = str(parsed["second_team"])
        feature = StructuredFootballFeature(
            feature_type="odds",
            impact_area="market_odds",
            feature_value=parsed,
            direction=float(parsed["direction"]),
            magnitude=float(parsed["magnitude"]),
            confidence=0.9,
            feature_confidence=0.9,
            match_id=str(event.get("id") or request.context.get("match_id") or ""),
            snapshot_at=snapshot_at,
            available_at=snapshot_at,
            prediction_deadline=request.prediction_deadline,
            leakage_risk="low" if commence_time and snapshot_at <= commence_time else "medium",
            extraction_method="the_odds_api_h2h",
            source_name=self.name,
            source_url=url,
            license_note="The Odds API response; use with a configured account key and respect plan limits.",
            source_provenance={
                "event_id": event.get("id"),
                "sport_key": sport_key,
                "regions": regions,
                "bookmaker_count": parsed["book_count"],
                "commence_time": event.get("commence_time"),
                "source_url": url,
            },
            rationale=(
                "The Odds API h2h market gives "
                f"{first_team} {float(parsed['first_prob']):.1%}, draw {float(parsed['draw_prob']):.1%}, "
                f"{second_team} {float(parsed['second_prob']):.1%} after overround adjustment."
            ),
        )
        claim = (
            f"The Odds API h2h 1X2 consensus: {first_team} {float(parsed['first_odds']):.2f}, "
            f"draw {float(parsed['draw_odds']):.2f}, {second_team} {float(parsed['second_odds']):.2f}; "
            f"probabilities first={float(parsed['first_prob']):.3f}, draw={float(parsed['draw_prob']):.3f}, "
            f"second={float(parsed['second_prob']):.3f}; bookmakers={int(parsed['book_count'])}; "
            f"commence_time={event.get('commence_time')}."
        )
        return [
            EvidenceItem(
                claim=claim,
                source=self.name,
                source_url=url,
                license_note=feature.license_note,
                source_provenance=feature.source_provenance,
                source_query=f"The Odds API {sport_key} h2h odds {first} {second}",
                evidence_stage="verified_candidate",
                raw_excerpt=claim,
                verifier_notes=["Parsed from The Odds API h2h odds endpoint."],
                published_at=snapshot_at,
                impact_area="market_odds",
                source_reliability=0.9,
                recency_score=0.95,
                corroboration_count=int(parsed["book_count"]),
                confidence=0.9,
                structured_features=[feature],
            )
        ]

    def _find_event(self, events: Any, first: str, second: str, match_id: Any) -> dict[str, Any] | None:
        if not isinstance(events, list):
            return None
        normalized_first = self._normalize(first)
        normalized_second = self._normalize(second)
        for event in events:
            if not isinstance(event, dict):
                continue
            if match_id and str(event.get("id")) == str(match_id):
                return event
            home = self._normalize(str(event.get("home_team") or ""))
            away = self._normalize(str(event.get("away_team") or ""))
            if self._team_matches(normalized_first, home) and self._team_matches(normalized_second, away):
                return event
            if self._team_matches(normalized_first, away) and self._team_matches(normalized_second, home):
                return event
        return None

    def _parse_event(self, event: dict[str, Any], first: str, second: str) -> dict[str, float | str] | None:
        home_team = str(event.get("home_team") or "")
        away_team = str(event.get("away_team") or "")
        home = self._normalize(home_team)
        away = self._normalize(away_team)
        first_is_home = self._team_matches(self._normalize(first), home)
        prices = self._average_prices(event, home_team, away_team)
        if not prices:
            return None
        home_odds = prices.get("home")
        draw_odds = prices.get("draw")
        away_odds = prices.get("away")
        if not home_odds or not draw_odds or not away_odds or min(home_odds, draw_odds, away_odds) <= 1:
            return None

        raw_home = 1 / home_odds
        raw_draw = 1 / draw_odds
        raw_away = 1 / away_odds
        total = raw_home + raw_draw + raw_away
        home_prob = raw_home / total
        draw_prob = raw_draw / total
        away_prob = raw_away / total
        first_prob = home_prob if first_is_home else away_prob
        second_prob = away_prob if first_is_home else home_prob
        first_odds = home_odds if first_is_home else away_odds
        second_odds = away_odds if first_is_home else home_odds
        edge = (first_prob - second_prob) / max(first_prob + second_prob, 0.001)
        return {
            "home_team": home_team,
            "away_team": away_team,
            "first_team": home_team if first_is_home else away_team,
            "second_team": away_team if first_is_home else home_team,
            "home_odds": home_odds,
            "draw_odds": draw_odds,
            "away_odds": away_odds,
            "first_odds": first_odds,
            "second_odds": second_odds,
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "first_prob": first_prob,
            "second_prob": second_prob,
            "overround": total - 1,
            "direction": 1.0 if edge >= 0 else -1.0,
            "magnitude": min(abs(edge), 1.0),
            "book_count": float(prices["book_count"]),
            "odds_type": "current_h2h",
            "bookmaker": "average_consensus",
        }

    def _average_prices(self, event: dict[str, Any], home_team: str, away_team: str) -> dict[str, float] | None:
        totals = {"home": 0.0, "draw": 0.0, "away": 0.0}
        counts = {"home": 0, "draw": 0, "away": 0}
        bookmaker_count = 0
        home = self._normalize(home_team)
        away = self._normalize(away_team)
        for bookmaker in event.get("bookmakers") or []:
            markets = bookmaker.get("markets") if isinstance(bookmaker, dict) else None
            market = next((item for item in markets or [] if item.get("key") == "h2h"), None)
            if not market:
                continue
            seen = False
            for outcome in market.get("outcomes") or []:
                name = self._normalize(str(outcome.get("name") or ""))
                try:
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                key = None
                if self._team_matches(home, name):
                    key = "home"
                elif self._team_matches(away, name):
                    key = "away"
                elif name == "draw":
                    key = "draw"
                if key:
                    totals[key] += price
                    counts[key] += 1
                    seen = True
            if seen:
                bookmaker_count += 1
        if not all(counts.values()):
            return None
        return {
            "home": totals["home"] / counts["home"],
            "draw": totals["draw"] / counts["draw"],
            "away": totals["away"] / counts["away"],
            "book_count": float(bookmaker_count),
        }

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None

    def _after(self, left: datetime, right: datetime) -> bool:
        try:
            return left > right
        except TypeError:
            return left.replace(tzinfo=None) > right.replace(tzinfo=None)

    def _team_matches(self, query: str, row_value: str) -> bool:
        return query == row_value or query in row_value or row_value in query

    def _normalize(self, value: str) -> str:
        normalized = value.lower().replace("&", "and")
        normalized = normalized.replace("united states", "usa")
        normalized = normalized.replace("czech republic", "czechia")
        normalized = normalized.replace("côte d'ivoire", "ivory coast")
        normalized = normalized.replace("cote d'ivoire", "ivory coast")
        return " ".join(normalized.split())

    def _gap(self, message: str) -> EvidenceItem:
        return EvidenceItem(
            claim=message,
            source=self.name,
            source_url="https://api.the-odds-api.com/v4",
            source_query="The Odds API h2h odds",
            evidence_stage="collection_gap",
            impact_area="market_odds",
            source_reliability=0.9,
            recency_score=0.0,
            confidence=0.0,
        )
