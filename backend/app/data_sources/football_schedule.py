from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from app.core.config import settings
from app.schemas import FootballScheduleMatch, FootballScheduleResponse


class FootballScheduleService:
    """Fetch same-day football fixtures from free/publicly accessible API paths.

    The Odds API is optional and user-keyed. The service returns an empty ready
    response with notes when the key is absent or no configured sport has events.
    """

    base_url = "https://api.the-odds-api.com/v4"

    async def today(
        self,
        target_date: date | None = None,
        tz_name: str = "Asia/Shanghai",
        days: int = 2,
    ) -> FootballScheduleResponse:
        tz = self._timezone(tz_name)
        days = max(min(days, 7), 1)
        local_date = target_date or datetime.now(tz).date()
        start_local = datetime.combine(local_date, time.min, tzinfo=tz)
        end_local = datetime.combine(local_date + timedelta(days=days - 1), time.max, tzinfo=tz)
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)
        notes: list[str] = []

        api_key = (settings.the_odds_api_key or "").strip()
        if not api_key:
            return FootballScheduleResponse(
                status="missing_api_key",
                date=local_date.isoformat(),
                timezone=tz_name,
                notes=["THE_ODDS_API_KEY is not configured; cannot fetch today's schedule."],
            )

        sport_keys = [
            item.strip()
            for item in settings.the_odds_api_schedule_sport_keys.split(",")
            if item.strip()
        ]
        matches: list[FootballScheduleMatch] = []
        async with httpx.AsyncClient(timeout=30) as client:
            for sport_key in sport_keys:
                params = {
                    "apiKey": api_key,
                    "regions": settings.the_odds_api_regions,
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                    "dateFormat": "iso",
                }
                url = f"{self.base_url}/sports/{sport_key}/odds"
                try:
                    response = await client.get(url, params=params)
                    if response.status_code == 404:
                        notes.append(f"{sport_key}: sport is unavailable or out of season.")
                        continue
                    response.raise_for_status()
                    events = response.json()
                except Exception as exc:
                    notes.append(f"{sport_key}: schedule fetch failed: {exc!r}")
                    continue
                if not isinstance(events, list):
                    notes.append(f"{sport_key}: unexpected response shape.")
                    continue
                for event in events:
                    parsed = self._parse_event(event, sport_key)
                    if parsed and start_utc <= parsed.commence_time.astimezone(timezone.utc) <= end_utc:
                        matches.append(parsed)

        matches.sort(key=lambda item: item.commence_time)
        return FootballScheduleResponse(
            status="ready" if matches else "empty",
            date=local_date.isoformat() if days == 1 else f"{local_date.isoformat()}..{(local_date + timedelta(days=days - 1)).isoformat()}",
            timezone=tz_name,
            matches=matches,
            notes=notes[:12],
        )

    def _timezone(self, tz_name: str):
        try:
            return ZoneInfo(tz_name)
        except ZoneInfoNotFoundError:
            if tz_name in {"Asia/Shanghai", "Asia/Chongqing", "China"}:
                return timezone(timedelta(hours=8), name="Asia/Shanghai")
            return timezone.utc

    def _parse_event(self, event: Any, sport_key: str) -> FootballScheduleMatch | None:
        if not isinstance(event, dict):
            return None
        match_id = str(event.get("id") or "").strip()
        home_team = str(event.get("home_team") or "").strip()
        away_team = str(event.get("away_team") or "").strip()
        commence_raw = event.get("commence_time")
        if not match_id or not home_team or not away_team or not commence_raw:
            return None
        try:
            commence_time = datetime.fromisoformat(str(commence_raw).replace("Z", "+00:00"))
        except ValueError:
            return None
        return FootballScheduleMatch(
            match_id=match_id,
            sport_key=sport_key,
            league=str(event.get("sport_title") or sport_key),
            commence_time=commence_time,
            home_team=home_team,
            away_team=away_team,
            odds=self._average_h2h(event, home_team, away_team),
            source="the_odds_api",
        )

    def _average_h2h(self, event: dict[str, Any], home_team: str, away_team: str) -> dict[str, Any]:
        totals = {"home": 0.0, "draw": 0.0, "away": 0.0}
        counts = {"home": 0, "draw": 0, "away": 0}
        bookmakers: list[str] = []
        home = self._normalize(home_team)
        away = self._normalize(away_team)
        for bookmaker in event.get("bookmakers") or []:
            if not isinstance(bookmaker, dict):
                continue
            market = next((item for item in bookmaker.get("markets") or [] if item.get("key") == "h2h"), None)
            if not market:
                continue
            used = False
            for outcome in market.get("outcomes") or []:
                if not isinstance(outcome, dict):
                    continue
                name = self._normalize(str(outcome.get("name") or ""))
                try:
                    price = float(outcome.get("price"))
                except (TypeError, ValueError):
                    continue
                key = None
                if name == "draw":
                    key = "draw"
                elif self._team_matches(home, name):
                    key = "home"
                elif self._team_matches(away, name):
                    key = "away"
                if key:
                    totals[key] += price
                    counts[key] += 1
                    used = True
            if used:
                bookmakers.append(str(bookmaker.get("title") or bookmaker.get("key") or "bookmaker"))
        if not all(counts.values()):
            return {"available": False, "bookmaker_count": len(bookmakers)}

        odds = {key: totals[key] / counts[key] for key in totals}
        raw = {key: 1 / value for key, value in odds.items()}
        total = sum(raw.values())
        probabilities = {key: raw[key] / total for key in raw}
        return {
            "available": True,
            "home_odds": round(odds["home"], 4),
            "draw_odds": round(odds["draw"], 4),
            "away_odds": round(odds["away"], 4),
            "home_probability": round(probabilities["home"], 4),
            "draw_probability": round(probabilities["draw"], 4),
            "away_probability": round(probabilities["away"], 4),
            "bookmaker_count": len(bookmakers),
            "bookmakers": bookmakers[:8],
        }

    def _team_matches(self, query: str, row_value: str) -> bool:
        return query == row_value or query in row_value or row_value in query

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().replace("&", "and").split())
