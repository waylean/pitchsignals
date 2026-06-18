from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from typing import Any

import httpx

from app.core.entities import infer_competitors
from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest, StructuredFootballFeature


class FootballDataFixturesOddsSource:
    """Public future-fixture 1X2 odds from Football-Data.co.uk.

    Football-Data publishes a fixtures CSV for upcoming club matches. It is a
    public/free source and usually includes 1X2 bookmaker columns such as B365H,
    B365D, B365A, MaxH/MaxD/MaxA, and AvgH/AvgD/AvgA.
    """

    name = "football_data_fixtures_odds"
    fixtures_url = "https://www.football-data.co.uk/fixtures.csv"

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if request.domain != "football":
            return []

        first, second = infer_competitors(request)
        home = self._normalize(str(request.context.get("home_team") or first or ""))
        away = self._normalize(str(request.context.get("away_team") or second or ""))
        if not home or not away:
            return [self._gap("Future fixture odds require two competitors.")]

        csv_url = str(request.context.get("football_data_fixtures_url") or self.fixtures_url)
        if not csv_url.startswith("https://www.football-data.co.uk/"):
            return [self._gap("Future fixture odds URL must be a Football-Data.co.uk URL.")]

        try:
            async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "ForecastIntelligence/0.1"}) as client:
                response = await client.get(csv_url)
                response.raise_for_status()
                response.encoding = "utf-8-sig"
        except Exception as exc:
            return [self._gap(f"Football-Data future fixtures odds collection failed: {exc}")]

        allow_past_fixtures = bool(request.context.get("allow_past_football_data_fixtures"))
        row = self._find_match(response.text, home, away, allow_past_fixtures=allow_past_fixtures)
        if not row:
            return [self._gap(f"No future Football-Data fixture row matched {home} vs {away}.")]

        odds = self._extract_odds(row)
        if not odds:
            return [self._gap("Matched Football-Data future fixture has no usable 1X2 odds columns.")]

        home_odds, draw_odds, away_odds, odds_prefix = odds
        implied = self._implied_probabilities(home_odds, draw_odds, away_odds)
        home_team = str(row.get("HomeTeam") or home)
        away_team = str(row.get("AwayTeam") or away)
        first_is_home = self._normalize(home_team) == home
        first_prob = implied["home"] if first_is_home else implied["away"]
        second_prob = implied["away"] if first_is_home else implied["home"]
        directional_value = (first_prob - second_prob) / max(first_prob + second_prob, 0.001)
        feature = StructuredFootballFeature(
            feature_type="future_fixture_1x2_odds",
            impact_area="market_odds",
            feature_value={
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": home_odds,
                "draw_odds": draw_odds,
                "away_odds": away_odds,
                "home_prob": implied["home"],
                "draw_prob": implied["draw"],
                "away_prob": implied["away"],
                "odds_prefix": odds_prefix,
                "source_url": csv_url,
            },
            direction=1.0 if directional_value >= 0 else -1.0,
            magnitude=min(abs(directional_value), 1.0),
            confidence=0.82,
            feature_confidence=0.82,
            extraction_method="football_data_fixtures_csv",
            source_name=self.name,
            source_url=csv_url,
            license_note=(
                "Public/free Football-Data.co.uk fixtures CSV; keep source attribution and "
                "do not redistribute cached data without checking site terms."
            ),
            source_provenance={
                "source_url": csv_url,
                "fixture_date": row.get("Date"),
                "fixture_time": row.get("Time"),
                "division": row.get("Div"),
                "odds_prefix": odds_prefix,
            },
            rationale=(
                f"Football-Data future fixture 1X2 odds imply {home_team} {implied['home']:.1%}, "
                f"draw {implied['draw']:.1%}, {away_team} {implied['away']:.1%}."
            ),
        )
        claim = (
            f"Football-Data future fixture 1X2 odds ({odds_prefix}): {home_team} {home_odds:.2f}, "
            f"draw {draw_odds:.2f}, {away_team} {away_odds:.2f}; probabilities home={implied['home']:.3f}, "
            f"draw={implied['draw']:.3f}, away={implied['away']:.3f}; fixture={row.get('Date')} {row.get('Time')}."
        )
        return [
            EvidenceItem(
                claim=claim,
                source=self.name,
                source_url=csv_url,
                license_note=feature.license_note,
                source_provenance=feature.source_provenance,
                source_query=f"Football-Data future fixtures odds {home} {away}",
                evidence_stage="verified_candidate",
                raw_excerpt=claim,
                verifier_notes=["Parsed from public Football-Data.co.uk fixtures CSV."],
                published_at=None,
                impact_area="market_odds",
                source_reliability=0.84,
                recency_score=0.82,
                corroboration_count=1,
                confidence=0.82,
                structured_features=[feature],
            )
        ]

    def _find_match(
        self,
        text: str,
        home: str,
        away: str,
        *,
        allow_past_fixtures: bool = False,
    ) -> dict[str, str] | None:
        rows = list(csv.DictReader(StringIO(text.lstrip("\ufeff"))))
        now = datetime.now()
        for row in rows:
            row_home = self._normalize(row.get("HomeTeam", ""))
            row_away = self._normalize(row.get("AwayTeam", ""))
            fixture_at = self._fixture_datetime(row)
            if fixture_at and not allow_past_fixtures and fixture_at < now:
                continue
            if self._team_matches(home, row_home) and self._team_matches(away, row_away):
                return row
        return None

    def _team_matches(self, query: str, row_value: str) -> bool:
        return query == row_value or query in row_value or row_value in query

    def _extract_odds(self, row: dict[str, str]) -> tuple[float, float, float, str] | None:
        for prefix in ["Avg", "Max", "B365", "PS", "BFD", "BMGM", "BV", "BW"]:
            try:
                home = float(row.get(f"{prefix}H") or "")
                draw = float(row.get(f"{prefix}D") or "")
                away = float(row.get(f"{prefix}A") or "")
            except ValueError:
                continue
            if min(home, draw, away) > 1:
                return home, draw, away, prefix
        return None

    def _implied_probabilities(self, home: float, draw: float, away: float) -> dict[str, float]:
        raw = {"home": 1 / home, "draw": 1 / draw, "away": 1 / away}
        total = sum(raw.values()) or 1
        return {key: value / total for key, value in raw.items()}

    def _date(self, value: str | None) -> datetime | None:
        if not value:
            return None
        for fmt in ["%d/%m/%Y", "%d/%m/%y"]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    def _fixture_datetime(self, row: dict[str, str]) -> datetime | None:
        parsed = self._date(row.get("Date"))
        if not parsed:
            return None
        time_value = (row.get("Time") or "").strip()
        if not time_value:
            return parsed
        try:
            hours, minutes = [int(part) for part in time_value.split(":", 1)]
        except ValueError:
            return parsed
        return parsed.replace(hour=hours, minute=minutes)

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().replace("&", "and").split())

    def _gap(self, message: str) -> EvidenceItem:
        return EvidenceItem(
            claim=message,
            source=self.name,
            source_url=self.fixtures_url,
            source_query="Football-Data.co.uk future fixtures CSV",
            evidence_stage="collection_gap",
            impact_area="market_odds",
            source_reliability=0.82,
            recency_score=0.0,
            confidence=0.0,
        )
