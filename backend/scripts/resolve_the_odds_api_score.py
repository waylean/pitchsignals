from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.entities import infer_competitors  # noqa: E402
from app.schemas import PredictionRequest  # noqa: E402


async def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a football match score through The Odds API scores endpoint.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--sport-key", default="soccer_fifa_world_cup")
    parser.add_argument("--days-from", type=int, default=3)
    parser.add_argument("--api-key", default=os.environ.get("THE_ODDS_API_KEY"))
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key or THE_ODDS_API_KEY is required")

    request = PredictionRequest(question=args.question, domain="football")
    first, second = infer_competitors(request)
    if not first or not second:
        raise SystemExit("Could not infer two competitors from question.")

    url = f"https://api.the-odds-api.com/v4/sports/{args.sport_key}/scores"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            url,
            params={"apiKey": args.api_key, "daysFrom": args.days_from, "dateFormat": "iso"},
        )
        response.raise_for_status()
        events = response.json()

    event = _find_event(events, first, second)
    if not event:
        print(json.dumps({"status": "missing", "first": first, "second": second}, ensure_ascii=False))
        return 1
    resolved = _resolve_event(event)
    print(json.dumps(resolved, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _find_event(events: Any, first: str, second: str) -> dict[str, Any] | None:
    if not isinstance(events, list):
        return None
    for event in events:
        if not isinstance(event, dict):
            continue
        home = _normalize(str(event.get("home_team") or ""))
        away = _normalize(str(event.get("away_team") or ""))
        first_norm = _normalize(first)
        second_norm = _normalize(second)
        if _team_matches(first_norm, home) and _team_matches(second_norm, away):
            return event
        if _team_matches(first_norm, away) and _team_matches(second_norm, home):
            return event
    return None


def _resolve_event(event: dict[str, Any]) -> dict[str, Any]:
    scores = event.get("scores") or []
    score_by_team = {str(item.get("name")): int(item.get("score")) for item in scores if isinstance(item, dict)}
    home_team = str(event.get("home_team") or "")
    away_team = str(event.get("away_team") or "")
    home_score = score_by_team.get(home_team)
    away_score = score_by_team.get(away_team)
    actual_outcome = None
    if home_score is not None and away_score is not None:
        if home_score > away_score:
            actual_outcome = "home_win"
        elif home_score < away_score:
            actual_outcome = "away_win"
        else:
            actual_outcome = "draw"
    return {
        "status": "ready" if actual_outcome else "unresolved",
        "actual_outcome": actual_outcome,
        "completed": bool(event.get("completed")),
        "home_team": home_team,
        "away_team": away_team,
        "home_score": home_score,
        "away_score": away_score,
        "commence_time": event.get("commence_time"),
        "last_update": event.get("last_update"),
        "source": "the_odds_api_scores",
        "resolved_at": datetime.utcnow().isoformat(),
    }


def _team_matches(query: str, row_value: str) -> bool:
    return query == row_value or query in row_value or row_value in query


def _normalize(value: str) -> str:
    normalized = value.lower().replace("&", "and")
    normalized = normalized.replace("czech republic", "czechia")
    normalized = normalized.replace("united states", "usa")
    return " ".join(normalized.split())


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
