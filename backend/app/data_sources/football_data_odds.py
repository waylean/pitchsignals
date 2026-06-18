from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from typing import Any

import httpx

from app.core.entities import infer_competitors
from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest


_LEAGUE_CODES = {
    "premier_league": "E0",
    "championship": "E1",
    "la_liga": "SP1",
    "bundesliga": "D1",
    "serie_a": "I1",
    "ligue_1": "F1",
}


class FootballDataOddsSource:
    """Public Football-Data.co.uk odds adapter.

    This is mainly useful for club fixtures and backtests. It requires either
    request.context["football_data_csv_url"] or request.context["football_data_league"].
    """

    name = "football_data_co_uk"
    base_url = "https://www.football-data.co.uk"

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if request.domain != "football":
            return []

        csv_url = self._csv_url(request.context)
        if not csv_url:
            return [self._gap("Football-Data odds require football_data_league or football_data_csv_url in context.")]

        first, second = infer_competitors(request)
        home = str(request.context.get("home_team") or first or "").lower()
        away = str(request.context.get("away_team") or second or "").lower()
        if not home or not away:
            return [self._gap("Football-Data odds require two competitors or home_team/away_team in context.")]

        try:
            async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "ForecastIntelligence/0.1"}) as client:
                response = await client.get(csv_url)
                response.raise_for_status()
        except Exception as exc:
            return [self._gap(f"Football-Data odds collection failed: {exc}")]

        match = self._find_match(response.text, home, away)
        if not match:
            return [self._gap(f"No matching Football-Data row found for {home} vs {away}.")]

        odds = self._extract_odds(match)
        if not odds:
            return [self._gap("Matching Football-Data row has no usable 1X2 odds columns.")]

        home_odds, draw_odds, away_odds = odds
        implied = self._implied_probabilities(home_odds, draw_odds, away_odds)
        claim = (
            f"Football-Data 1X2 odds: {match.get('HomeTeam')} {home_odds:.2f}, "
            f"draw {draw_odds:.2f}, {match.get('AwayTeam')} {away_odds:.2f}; "
            f"overround-adjusted probabilities home={implied['home']:.3f}, "
            f"draw={implied['draw']:.3f}, away={implied['away']:.3f}."
        )
        return [
            EvidenceItem(
                claim=claim,
                source=self.name,
                source_url=csv_url,
                source_query=f"Football-Data odds {home} {away}",
                evidence_stage="verified_candidate",
                raw_excerpt=claim,
                verifier_notes=["Parsed from public Football-Data.co.uk CSV."],
                published_at=self._date(match.get("Date")),
                impact_area="market_odds",
                source_reliability=0.84,
                recency_score=0.65,
                corroboration_count=1,
                confidence=0.82,
            )
        ]

    def _csv_url(self, context: dict[str, Any]) -> str | None:
        explicit = context.get("football_data_csv_url")
        if isinstance(explicit, str) and explicit.startswith("https://www.football-data.co.uk/"):
            return explicit

        league = context.get("football_data_league")
        if not isinstance(league, str):
            return None
        code = _LEAGUE_CODES.get(league.lower(), league.upper())
        season = str(context.get("football_data_season") or self._current_season_code())
        return f"{self.base_url}/mmz4281/{season}/{code}.csv"

    def _current_season_code(self) -> str:
        now = datetime.utcnow()
        start_year = now.year - 1 if now.month < 7 else now.year
        return f"{str(start_year)[-2:]}{str(start_year + 1)[-2:]}"

    def _find_match(self, text: str, home: str, away: str) -> dict[str, str] | None:
        rows = list(csv.DictReader(StringIO(text)))
        for row in reversed(rows):
            row_home = self._normalize(row.get("HomeTeam", ""))
            row_away = self._normalize(row.get("AwayTeam", ""))
            if home in row_home and away in row_away:
                return row
        return None

    def _extract_odds(self, row: dict[str, str]) -> tuple[float, float, float] | None:
        for prefix in ["Avg", "B365", "Max"]:
            try:
                home = float(row.get(f"{prefix}H") or "")
                draw = float(row.get(f"{prefix}D") or "")
                away = float(row.get(f"{prefix}A") or "")
            except ValueError:
                continue
            if min(home, draw, away) > 1:
                return home, draw, away
        return None

    def _implied_probabilities(self, home: float, draw: float, away: float) -> dict[str, float]:
        raw = {"home": 1 / home, "draw": 1 / draw, "away": 1 / away}
        total = sum(raw.values())
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

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().split())

    def _gap(self, message: str) -> EvidenceItem:
        return EvidenceItem(
            claim=message,
            source=self.name,
            source_url=self.base_url,
            source_query="Football-Data.co.uk odds CSV",
            evidence_stage="collection_gap",
            impact_area="market_odds",
            source_reliability=0.82,
            recency_score=0.0,
            confidence=0.0,
        )
