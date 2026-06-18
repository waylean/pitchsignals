from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from io import StringIO
from pathlib import Path

import httpx

from app.backtesting.records import (
    DatasetManifest,
    FeatureSnapshot,
    FootballBacktestDataset,
    FootballMatchRecord,
)


FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"

FOOTBALL_DATA_LEAGUES = {
    "premier_league": "E0",
    "championship": "E1",
    "la_liga": "SP1",
    "bundesliga": "D1",
    "serie_a": "I1",
    "ligue_1": "F1",
}


async def fetch_text_cached(url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(
        timeout=45,
        follow_redirects=True,
        headers={"User-Agent": "ForecastIntelligence/0.1"},
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


async def load_football_data_dataset(
    cache_dir: Path,
    league: str = "premier_league",
    season: str = "2324",
) -> FootballBacktestDataset:
    code = FOOTBALL_DATA_LEAGUES.get(league, league.upper())
    url = f"{FOOTBALL_DATA_BASE_URL}/{season}/{code}.csv"
    cache_path = cache_dir / "football_data" / season / f"{code}.csv"
    text = await fetch_text_cached(url, cache_path)
    rows = list(csv.DictReader(StringIO(text.lstrip("\ufeff"))))

    records: list[FootballMatchRecord] = []
    gaps: list[str] = []
    for index, row in enumerate(rows, start=1):
        record = _football_data_row_to_record(row, index, league, season, code)
        if record:
            records.append(record)
        else:
            gaps.append(f"Skipped row {index}: missing result, date, teams, or 1X2 odds.")

    manifest = DatasetManifest(
        dataset_id=f"football_data_{league}_{season}",
        source_url=url,
        license_note=(
            "Football-Data.co.uk public CSV. Do not redistribute the full dataset in-repo; "
            "cache locally and attribute source."
        ),
        cache_path=str(cache_path),
        competition=league,
        season=season,
        case_count=len(records),
        dataset_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        snapshot_policy="downloaded_csv_cache",
        horizon_policy="market_snapshot_unknown_or_closing_proxy",
        license_tier="public_free_attributed",
        redistribution_allowed=False,
        coverage={
            "rows": len(rows),
            "records_with_1x2_odds": sum(1 for record in records if record.has_1x2_odds),
            "leakage_issues": sum(len(record.leakage_issues()) for record in records),
        },
        gaps=gaps[:20],
    )
    return FootballBacktestDataset(manifest=manifest, records=records)


def _football_data_row_to_record(
    row: dict[str, str],
    index: int,
    league: str,
    season: str,
    code: str,
) -> FootballMatchRecord | None:
    try:
        home_team = row["HomeTeam"].strip()
        away_team = row["AwayTeam"].strip()
        home_goals = int(row["FTHG"])
        away_goals = int(row["FTAG"])
    except (KeyError, ValueError):
        return None
    event_time = _parse_football_data_date(row.get("Date"))
    if not event_time or not home_team or not away_team:
        return None
    odds = _extract_football_data_odds(row)
    if not odds:
        return None
    snapshot_id = f"football_data:{code}:{season}:{index:04d}:odds"
    odds_snapshot = FeatureSnapshot(
        snapshot_id=snapshot_id,
        source="football-data.co.uk",
        snapshot_type="market_snapshot_unknown_or_closing_proxy",
        available_at=None,
        confidence=0.55,
        values={
            "selected_price_policy": odds.get("source"),
            "home": odds.get("home"),
            "draw": odds.get("draw"),
            "away": odds.get("away"),
            "overround": odds.get("overround"),
            "bookmaker_count": odds.get("bookmaker_count"),
            "odds_source_columns": odds.get("odds_source_columns"),
        },
    )
    return FootballMatchRecord(
        match_id=f"{code}-{season}-{index:04d}",
        dataset_id=f"football_data_{league}_{season}",
        competition=league,
        season=season,
        event_time=event_time,
        home_team=home_team,
        away_team=away_team,
        home_goals=home_goals,
        away_goals=away_goals,
        odds=odds,
        prediction_deadline=event_time,
        odds_as_of=None,
        odds_available_at=None,
        feature_available_at=None,
        source_snapshot_id=snapshot_id,
        odds_snapshot_type="market_snapshot_unknown_or_closing_proxy",
        odds_snapshot=odds_snapshot,
        source_row=row,
    )


def _extract_football_data_odds(row: dict[str, str]) -> dict[str, float] | None:
    for prefix in ["Avg", "B365", "Max"]:
        try:
            home = float(row.get(f"{prefix}H") or "")
            draw = float(row.get(f"{prefix}D") or "")
            away = float(row.get(f"{prefix}A") or "")
        except ValueError:
            continue
        if min(home, draw, away) > 1:
            implied = (1 / home) + (1 / draw) + (1 / away)
            return {
                "home": home,
                "draw": draw,
                "away": away,
                "source": prefix,
                "overround": round(implied - 1, 6),
                "bookmaker_count": 1 if prefix == "B365" else 0,
                "odds_source_columns": f"{prefix}H,{prefix}D,{prefix}A",
            }
    return None


def _parse_football_data_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ["%d/%m/%Y", "%d/%m/%y"]:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None
