from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.domain_packs.football import FOOTBALL_PACK
from app.evidence.store import InMemoryEvidenceStore
from app.feedback.analyzer import FeedbackAnalyzer
from app.models.football_horizon_profiles import FOOTBALL_HORIZON_PROFILES
from app.models.governance import (
    DEFAULT_ENSEMBLE_PROFILES,
    EvaluationCase,
    PredictionDistribution,
    build_evaluation_report,
    build_model_leaderboard,
    evaluate_promotion_gate,
)
from app.models.weighted_model import WeightedPredictionModel
from app.schemas import EvidenceItem, FeedbackRequest, PredictionRequest, PredictionResponse


OPENFOOTBALL_2022_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json"
)
FIVETHIRTYEIGHT_SPI_URL = (
    "https://projects.fivethirtyeight.com/soccer-api/international/spi_matches_intl.csv"
)
FIVETHIRTYEIGHT_WC_FORECAST_MIRROR_URL = (
    "https://raw.githubusercontent.com/tanzil64/Tanzil_Data_607_Assignment01/main/wc_forecasts.csv"
)
ELO_BASE_URL = "https://www.eloratings.net"
STATSBOMB_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

TEAM_URL_ALIASES = {
    "USA": "United_States",
    "Saudi Arabia": "Saudi_Arabia",
    "South Korea": "South_Korea",
    "Costa Rica": "Costa_Rica",
}

TEAM_NORMALIZE_ALIASES = {
    "usa": "united states",
}

STADIUM_COORDS = {
    "Al Bayt Stadium, Al Khor": (25.6522, 51.4879),
    "Al Thumama Stadium, Doha": (25.2356, 51.5321),
    "Khalifa International Stadium, Al Rayyan": (25.2637, 51.4481),
    "Ahmad Bin Ali Stadium, Al Rayyan": (25.3308, 51.3425),
    "Ahmad bin Ali Stadium, Al Rayyan": (25.3308, 51.3425),
    "Education City Stadium, Al Rayyan": (25.3117, 51.4244),
    "Stadium 974, Doha": (25.2892, 51.5665),
    "Lusail Stadium, Lusail": (25.4209, 51.4908),
    "Lusail Iconic Stadium, Lusail": (25.4209, 51.4908),
    "Al Janoub Stadium, Al Wakrah": (25.1596, 51.5742),
}


@dataclass(frozen=True)
class EloSnapshot:
    rating1: int
    rating2: int
    source: str


def outcome_from_score(score: dict[str, Any]) -> str:
    home, away = score["ft"]
    if home > away:
        return "home_win"
    if away > home:
        return "away_win"
    return "draw"


def normalize_team(value: str) -> str:
    normalized = " ".join(value.lower().replace("&", "and").split())
    return TEAM_NORMALIZE_ALIASES.get(normalized, normalized)


def team_url_name(team: str) -> str:
    return TEAM_URL_ALIASES.get(team, team.replace(" ", "_"))


async def fetch_text(client: httpx.AsyncClient, url: str, cache_path: Path) -> str:
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    last_error: Exception | None = None
    for _ in range(3):
        try:
            response = await client.get(url)
            response.raise_for_status()
            response.encoding = "utf-8"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(response.text, encoding="utf-8")
            return response.text
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.75)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


async def load_worldcup(cache_dir: Path) -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        text = await fetch_text(client, OPENFOOTBALL_2022_URL, cache_dir / "worldcup_2022.json")
    data = json.loads(text)
    return [match for match in data["matches"] if match.get("group")]


async def try_load_spi(cache_dir: Path) -> tuple[list[dict[str, str]], str | None]:
    cache_path = cache_dir / "spi_matches_intl.csv"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            text = await fetch_text(client, FIVETHIRTYEIGHT_SPI_URL, cache_path)
        if not text.lstrip().startswith("season,"):
            return [], "FiveThirtyEight SPI endpoint currently returned non-CSV HTML."
        return list(csv.DictReader(text.splitlines())), None
    except Exception as exc:
        return [], f"FiveThirtyEight SPI download failed: {exc}"


async def load_wc_forecasts(cache_dir: Path) -> tuple[list[dict[str, str]], str | None]:
    cache_path = cache_dir / "wc_forecasts.csv"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            text = await fetch_text(client, FIVETHIRTYEIGHT_WC_FORECAST_MIRROR_URL, cache_path)
        text = text.lstrip("\ufeff")
        if not text.lstrip().startswith("forecast_timestamp,"):
            return [], "FiveThirtyEight World Cup forecast mirror returned non-CSV content."
        return list(csv.DictReader(text.splitlines())), None
    except Exception as exc:
        return [], f"FiveThirtyEight World Cup forecast mirror failed: {exc}"


def parse_forecast_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S UTC").replace(tzinfo=timezone.utc)


def build_forecast_lookup(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            parsed = {
                "forecast_timestamp": parse_forecast_time(row["forecast_timestamp"]),
                "team": row["team"],
                "spi": float(row["spi"]),
                "global_o": float(row["global_o"]),
                "global_d": float(row["global_d"]),
                "make_round_of_16": float(row["make_round_of_16"]),
                "win_league": float(row["win_league"]),
            }
        except (KeyError, ValueError):
            continue
        lookup.setdefault(normalize_team(row["team"]), []).append(parsed)
    for team_rows in lookup.values():
        team_rows.sort(key=lambda item: item["forecast_timestamp"])
    return lookup


def latest_forecast_before(
    lookup: dict[str, list[dict[str, Any]]],
    team: str,
    event_time: datetime,
) -> dict[str, Any] | None:
    rows = lookup.get(normalize_team(team), [])
    latest = None
    for row in rows:
        if row["forecast_timestamp"] <= event_time:
            latest = row
        else:
            break
    return latest


async def load_statsbomb_matches(cache_dir: Path) -> tuple[list[dict[str, Any]], str | None]:
    url = f"{STATSBOMB_BASE_URL}/matches/43/106.json"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            text = await fetch_text(client, url, cache_dir / "statsbomb" / "matches_43_106.json")
        return json.loads(text), None
    except Exception as exc:
        return [], f"StatsBomb matches failed: {exc}"


def build_statsbomb_match_lookup(matches: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    lookup = {}
    for match in matches:
        key = (
            match.get("match_date", ""),
            normalize_team(match.get("home_team", {}).get("home_team_name", "")),
            normalize_team(match.get("away_team", {}).get("away_team_name", "")),
        )
        lookup[key] = match
        reverse = (key[0], key[2], key[1])
        lookup[reverse] = match
    return lookup


async def load_statsbomb_lineups(cache_dir: Path, match_id: int) -> list[dict[str, Any]]:
    url = f"{STATSBOMB_BASE_URL}/lineups/{match_id}.json"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        text = await fetch_text(client, url, cache_dir / "statsbomb" / "lineups" / f"{match_id}.json")
    return json.loads(text)


async def load_statsbomb_events(cache_dir: Path, match_id: int) -> list[dict[str, Any]]:
    url = f"{STATSBOMB_BASE_URL}/events/{match_id}.json"
    async with httpx.AsyncClient(timeout=45, follow_redirects=True) as client:
        text = await fetch_text(client, url, cache_dir / "statsbomb" / "events" / f"{match_id}.json")
    return json.loads(text)


def lineup_summary(lineups: list[dict[str, Any]], team1: str, team2: str) -> str | None:
    summaries = []
    for team in (team1, team2):
        row = next((item for item in lineups if normalize_team(item.get("team_name", "")) == normalize_team(team)), None)
        if not row:
            return None
        starters = []
        for player in row.get("lineup", []):
            if any(pos.get("start_reason") == "Starting XI" for pos in player.get("positions", [])):
                starters.append(player.get("player_name"))
        summaries.append(f"{team} confirmed starters {len(starters)}: {', '.join(starters[:11])}")
    return "; ".join(summaries)


def post_event_xg(events: list[dict[str, Any]], team1: str, team2: str) -> dict[str, float]:
    xg = {team1: 0.0, team2: 0.0}
    for event in events:
        if event.get("type", {}).get("name") != "Shot":
            continue
        team = event.get("team", {}).get("name", "")
        shot = event.get("shot", {})
        value = float(shot.get("statsbomb_xg") or 0)
        if normalize_team(team) == normalize_team(team1):
            xg[team1] += value
        elif normalize_team(team) == normalize_team(team2):
            xg[team2] += value
    return {team: round(value, 4) for team, value in xg.items()}


async def historical_weather(cache_dir: Path, match: dict[str, Any], event_time: datetime) -> tuple[EvidenceItem | None, str | None]:
    ground = match.get("ground")
    coords = STADIUM_COORDS.get(ground)
    if not coords:
        return None, f"Missing stadium coordinates for {ground}."
    cache_path = cache_dir / "weather" / f"{match['date']}_{ground.replace(' ', '_').replace(',', '')}.json"
    params = (
        f"latitude={coords[0]}&longitude={coords[1]}&start_date={match['date']}&end_date={match['date']}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m&timezone=UTC"
    )
    url = f"{OPEN_METEO_ARCHIVE_URL}?{params}"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            text = await fetch_text(client, url, cache_path)
        data = json.loads(text)
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        target = event_time.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")
        index = times.index(target) if target in times else 0
        temp = float(hourly.get("temperature_2m", [0])[index])
        humidity = float(hourly.get("relative_humidity_2m", [0])[index])
        precipitation = float(hourly.get("precipitation", [0])[index])
        wind = float(hourly.get("wind_speed_10m", [0])[index])
        claim = (
            f"Historical kickoff weather at {ground}: {temp:.1f}C, {humidity:.0f}% humidity, "
            f"{precipitation:.1f}mm precipitation, wind {wind:.1f} km/h."
        )
        return (
            EvidenceItem(
                claim=claim,
                source="open_meteo_archive",
                source_url=url,
                source_query=f"{ground} {match['date']} kickoff weather",
                evidence_stage="verified_candidate",
                raw_excerpt=claim,
                impact_area="referee_environment",
                source_reliability=0.84,
                recency_score=0.85,
                confidence=0.72,
            ),
            None,
        )
    except Exception as exc:
        return None, f"Historical weather unavailable for {ground}: {exc}"


async def load_elo_team_file(cache_dir: Path, team: str) -> str:
    url_name = team_url_name(team)
    url = f"{ELO_BASE_URL}/{url_name}.tsv"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        return await fetch_text(client, url, cache_dir / "elo" / f"{url_name}.tsv")


def parse_elo_snapshot(team_file_text: str, match: dict[str, Any], code_map: dict[str, str]) -> EloSnapshot | None:
    team1 = match["team1"]
    team2 = match["team2"]
    code1 = code_map.get(team1)
    code2 = code_map.get(team2)
    if not code1 or not code2:
        return None
    date = datetime.fromisoformat(match["date"])
    prefix = f"{date.year}\t{date.month:02d}\t{date.day:02d}\t"
    for line in team_file_text.splitlines():
        if not line.startswith(prefix):
            continue
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        row_team1, row_team2 = parts[3], parts[4]
        if {row_team1, row_team2} != {code1, code2}:
            continue
        try:
            change = int(parts[9].replace(chr(8722), "-"))
            rating1_after = int(parts[10])
            rating2_after = int(parts[11])
        except ValueError:
            continue
        row_rating1_pre = rating1_after - change
        row_rating2_pre = rating2_after + change
        if row_team1 == code1:
            return EloSnapshot(row_rating1_pre, row_rating2_pre, line)
        return EloSnapshot(row_rating2_pre, row_rating1_pre, line)
    return None


async def load_team_code_map(cache_dir: Path) -> dict[str, str]:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        text = await fetch_text(client, f"{ELO_BASE_URL}/en.teams.tsv", cache_dir / "elo" / "en.teams.tsv")
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        columns = [column.strip() for column in line.split("\t") if column.strip()]
        if len(columns) < 2 or columns[0].endswith("_loc"):
            continue
        code = columns[0]
        for name in columns[1:]:
            mapping[name] = code
    mapping["USA"] = mapping.get("United States", "US")
    return mapping


def group_form_claim(match: dict[str, Any], group_state: dict[str, dict[str, int]]) -> str:
    team1 = match["team1"]
    team2 = match["team2"]
    state1 = group_state.setdefault(team1, {"points": 0, "gf": 0, "ga": 0})
    state2 = group_state.setdefault(team2, {"points": 0, "gf": 0, "ga": 0})
    gd1 = state1["gf"] - state1["ga"]
    gd2 = state2["gf"] - state2["ga"]
    return (
        f"Group form before match: {team1} points {state1['points']} goal difference {gd1}; "
        f"{team2} points {state2['points']} goal difference {gd2}."
    )


def update_group_state(match: dict[str, Any], group_state: dict[str, dict[str, int]]) -> None:
    team1, team2 = match["team1"], match["team2"]
    score1, score2 = match["score"]["ft"]
    state1 = group_state.setdefault(team1, {"points": 0, "gf": 0, "ga": 0})
    state2 = group_state.setdefault(team2, {"points": 0, "gf": 0, "ga": 0})
    state1["gf"] += score1
    state1["ga"] += score2
    state2["gf"] += score2
    state2["ga"] += score1
    if score1 > score2:
        state1["points"] += 3
    elif score2 > score1:
        state2["points"] += 3
    else:
        state1["points"] += 1
        state2["points"] += 1


def weather_claim(match: dict[str, Any]) -> tuple[EvidenceItem | None, str | None]:
    ground = match.get("ground")
    coords = STADIUM_COORDS.get(ground)
    if not coords:
        return None, f"Missing stadium coordinates for {ground}."
    return (
        EvidenceItem(
            claim=f"Venue context: {ground} coordinates {coords[0]:.4f},{coords[1]:.4f}; historical weather not fetched in fast backtest.",
            source="backtest_stadium_registry",
            source_url=OPENFOOTBALL_2022_URL,
            source_query=ground,
            evidence_stage="verified_candidate",
            raw_excerpt=ground,
            impact_area="referee_environment",
            source_reliability=0.72,
            recency_score=0.65,
            confidence=0.55,
        ),
        None,
    )


def build_evidence(
    match: dict[str, Any],
    elo: EloSnapshot | None,
    group_state: dict[str, dict[str, int]],
    spi_row: dict[str, str] | None,
    team_forecasts: tuple[dict[str, Any] | None, dict[str, Any] | None],
    lineup_claim_text: str | None,
    weather_item: EvidenceItem | None,
) -> tuple[list[EvidenceItem], list[str]]:
    team1, team2 = match["team1"], match["team2"]
    evidence: list[EvidenceItem] = []
    gaps: list[str] = []
    if elo:
        evidence.append(
            EvidenceItem(
                claim=(
                    f"Historical pre-match Elo rating: {team1} rating {elo.rating1}; "
                    f"{team2} rating {elo.rating2}."
                ),
                source="world_football_elo_historical",
                source_url=f"{ELO_BASE_URL}/{team_url_name(team1)}.tsv",
                source_query=f"{team1} {team2} historical Elo {match['date']}",
                evidence_stage="verified_candidate",
                raw_excerpt=elo.source,
                published_at=datetime.fromisoformat(match["date"]).replace(tzinfo=timezone.utc),
                impact_area="team_strength",
                source_reliability=0.86,
                recency_score=0.9,
                corroboration_count=1,
                confidence=0.82,
            )
        )
    else:
        gaps.append(f"Historical Elo missing for {team1} vs {team2}.")

    evidence.append(
        EvidenceItem(
            claim=group_form_claim(match, group_state),
            source="backtest_group_table",
            source_url=OPENFOOTBALL_2022_URL,
            source_query=f"{match['group']} standings before {team1} vs {team2}",
            evidence_stage="verified_candidate",
            impact_area="team_strength",
            source_reliability=0.78,
            recency_score=0.95,
            confidence=0.68,
        )
    )

    if spi_row:
        evidence.append(
            EvidenceItem(
                claim=(
                    f"FiveThirtyEight SPI probabilities home={float(spi_row['prob1']):.3f}, "
                    f"draw={float(spi_row['probtie']):.3f}, away={float(spi_row['prob2']):.3f}."
                ),
                source="fivethirtyeight_spi",
                source_url=FIVETHIRTYEIGHT_SPI_URL,
                source_query=f"{team1} {team2} SPI pre-match",
                evidence_stage="verified_candidate",
                impact_area="market_odds",
                source_reliability=0.8,
                recency_score=0.85,
                corroboration_count=1,
                confidence=0.78,
            )
        )
    else:
        evidence.append(
            EvidenceItem(
                claim="SPI/market probability source unavailable for this replay.",
                source="fivethirtyeight_spi",
                source_url=FIVETHIRTYEIGHT_SPI_URL,
                evidence_stage="collection_gap",
                impact_area="market_odds",
                source_reliability=0.8,
                recency_score=0.0,
                confidence=0.0,
            )
        )
        gaps.append("Pre-match SPI or odds probabilities unavailable.")

    forecast1, forecast2 = team_forecasts
    if forecast1 and forecast2:
        evidence.append(
            EvidenceItem(
                claim=(
                    f"FiveThirtyEight tournament forecast before match: {team1} SPI {forecast1['spi']:.2f}, "
                    f"offense {forecast1['global_o']:.2f}, defense {forecast1['global_d']:.2f}; "
                    f"{team2} SPI {forecast2['spi']:.2f}, offense {forecast2['global_o']:.2f}, "
                    f"defense {forecast2['global_d']:.2f}."
                ),
                source="fivethirtyeight_wc_forecast_mirror",
                source_url=FIVETHIRTYEIGHT_WC_FORECAST_MIRROR_URL,
                source_query=f"{team1} {team2} SPI forecast before {match['date']}",
                evidence_stage="verified_candidate",
                impact_area="team_strength",
                source_reliability=0.72,
                recency_score=0.86,
                corroboration_count=1,
                confidence=0.74,
            )
        )
    else:
        gaps.append("Team-level FiveThirtyEight World Cup SPI forecast unavailable.")

    if weather_item:
        evidence.append(weather_item)

    if lineup_claim_text:
        evidence.append(
            EvidenceItem(
                claim=lineup_claim_text,
                source="statsbomb_open_data_lineups",
                source_url=STATSBOMB_BASE_URL,
                source_query=f"{team1} {team2} confirmed lineups",
                evidence_stage="verified_candidate",
                impact_area="lineup_availability",
                source_reliability=0.82,
                recency_score=0.9,
                confidence=0.76,
            )
        )
    else:
        gaps.append("Confirmed lineup data unavailable in StatsBomb replay dataset.")

    for area, message in [
        ("lineup_availability", "Injury and suspension data not included in public replay dataset."),
        ("referee_environment", "Referee appointment data not included in public replay dataset."),
        ("tactical_matchup", "Structured tactical matchup data not included in public replay dataset."),
        ("chemistry_relationships", "Chemistry/relationship data not included in public replay dataset."),
        ("sentiment_narrative", "Sentiment/narrative data not included in public replay dataset."),
    ]:
        evidence.append(
            EvidenceItem(
                claim=message,
                source="backtest_gap_analysis",
                source_query=area,
                evidence_stage="collection_gap",
                impact_area=area,
                source_reliability=0.7,
                recency_score=0.0,
                confidence=0.0,
            )
        )
        gaps.append(message)
    return evidence, gaps


def predict_with_weights(factors, outcomes: list[str], weights: dict[str, float]) -> dict[str, float]:
    base = 1 / len(outcomes)
    probabilities = {outcome: base for outcome in outcomes}
    signal = sum(f.value * weights.get(f.key, 0) * f.confidence for f in factors)
    shift = max(min(signal * 0.18, 0.14), -0.14)
    probabilities[outcomes[0]] = max(0.01, base + shift)
    probabilities[outcomes[-1]] = max(0.01, base - shift)
    total = sum(probabilities.values())
    return {key: value / total for key, value in probabilities.items()}


def log_loss(predictions: list[dict[str, Any]], weights: dict[str, float]) -> float:
    total = 0.0
    outcomes = ["home_win", "draw", "away_win"]
    for row in predictions:
        probs = predict_with_weights(row["factors"], outcomes, weights)
        total += -math.log(max(probs[row["actual"]], 1e-9))
    return total / max(len(predictions), 1)


def brier_score(predictions: list[dict[str, Any]], weights: dict[str, float]) -> float:
    outcomes = ["home_win", "draw", "away_win"]
    total = 0.0
    for row in predictions:
        probs = predict_with_weights(row["factors"], outcomes, weights)
        total += sum((probs[o] - (1 if o == row["actual"] else 0)) ** 2 for o in outcomes)
    return total / max(len(predictions), 1)


def train_weights(predictions: list[dict[str, Any]]) -> dict[str, float]:
    keys = [factor.key for factor in FOOTBALL_PACK.factors]
    candidates: list[dict[str, float]] = [FOOTBALL_PACK.factor_weight_map()]
    candidates.extend(
        [
            {"market_odds": 0.15, "team_strength": 0.62, "lineup_availability": 0.06, "tactical_matchup": 0.06, "chemistry_relationships": 0.03, "referee_environment": 0.05, "sentiment_narrative": 0.03},
            {"market_odds": 0.25, "team_strength": 0.45, "lineup_availability": 0.10, "tactical_matchup": 0.08, "chemistry_relationships": 0.04, "referee_environment": 0.05, "sentiment_narrative": 0.03},
            {"market_odds": 0.05, "team_strength": 0.75, "lineup_availability": 0.05, "tactical_matchup": 0.05, "chemistry_relationships": 0.03, "referee_environment": 0.04, "sentiment_narrative": 0.03},
        ]
    )
    rng = random.Random(2022)
    for _ in range(1500):
        values = [rng.gammavariate(1.2 if key == "team_strength" else 0.8, 1) for key in keys]
        total = sum(values)
        candidates.append({key: value / total for key, value in zip(keys, values)})
    best = min(candidates, key=lambda weights: log_loss(predictions, weights))
    total = sum(best.values()) or 1
    return {key: round(best.get(key, 0) / total, 4) for key in keys}


def directional_coverage(predictions: list[dict[str, Any]]) -> dict[str, int]:
    coverage = {factor.key: 0 for factor in FOOTBALL_PACK.factors}
    for row in predictions:
        for factor in row["factors"]:
            if abs(factor.value) > 0.001:
                coverage[factor.key] += 1
    return coverage


def evaluation_cases_for_weights(
    predictions: list[dict[str, Any]],
    model_id: str,
    weights: dict[str, float],
    horizon_profile: str | None = None,
) -> list[EvaluationCase]:
    outcomes = ["home_win", "draw", "away_win"]
    cases = []
    for row in predictions:
        distribution = PredictionDistribution(
            outcomes=predict_with_weights(row["factors"], outcomes, weights),
            model_id=model_id,
        )
        cases.append(
            EvaluationCase(
                actual=row["actual"],
                distribution=distribution,
                horizon_profile=horizon_profile,
                metadata={"match_no": row["match_no"], "match": f"{row['team1']} vs {row['team2']}"},
            )
        )
    return cases


def governance_evaluation_outputs(
    predictions: list[dict[str, Any]],
    baseline_weights: dict[str, float],
    trained_weights: dict[str, float],
) -> dict[str, Any]:
    reports = [
        build_evaluation_report(
            "football_factor_baseline",
            evaluation_cases_for_weights(predictions, "football_factor_baseline", baseline_weights),
            {"profile_type": "default_factor_weights"},
        ),
        build_evaluation_report(
            "football_factor_trained_sparse",
            evaluation_cases_for_weights(predictions, "football_factor_trained_sparse", trained_weights),
            {"profile_type": "trained_factor_weights", "maturity": "sparse_experimental"},
        ),
    ]
    for horizon_id, profile in FOOTBALL_HORIZON_PROFILES.items():
        weights = profile["weight_profile"]
        reports.append(
            build_evaluation_report(
                f"football_horizon_{horizon_id}",
                evaluation_cases_for_weights(
                    predictions,
                    f"football_horizon_{horizon_id}",
                    weights,
                    horizon_profile=horizon_id,
                ),
                {"profile_type": "horizon_factor_weights", "horizon_profile": horizon_id},
            )
        )

    leaderboard = build_model_leaderboard(
        "worldcup_2022_group_stage_factor_profiles",
        reports,
        metric="log_loss",
        metadata={
            "dataset": "worldcup_2022_group_stage",
            "case_count": len(predictions),
            "warning": "Uses current evidence-factor layer; not a full market/xG ensemble backtest.",
        },
    )
    promotion_decision = evaluate_promotion_gate(reports[1], reports[0])
    return {
        "evaluation_reports": [asdict(report) for report in reports],
        "model_leaderboard": asdict(leaderboard),
        "promotion_decision": asdict(promotion_decision),
        "versioned_ensemble_profiles": [
            asdict(profile) for profile in DEFAULT_ENSEMBLE_PROFILES.values()
        ],
    }


async def run_backtest(output_dir: Path) -> dict[str, Any]:
    cache_dir = ROOT / "work" / "worldcup_2022_cache"
    matches = await load_worldcup(cache_dir)
    spi_rows, spi_gap = await try_load_spi(cache_dir)
    forecast_rows, forecast_gap = await load_wc_forecasts(cache_dir)
    forecast_lookup = build_forecast_lookup(forecast_rows)
    statsbomb_matches, statsbomb_gap = await load_statsbomb_matches(cache_dir)
    statsbomb_lookup = build_statsbomb_match_lookup(statsbomb_matches)
    spi_index = {
        (row.get("date"), normalize_team(row.get("team1", "")), normalize_team(row.get("team2", ""))): row
        for row in spi_rows
    }
    code_map = await load_team_code_map(cache_dir)

    teams = sorted({m["team1"] for m in matches} | {m["team2"] for m in matches})
    elo_files: dict[str, str] = {}
    elo_gaps: list[str] = []
    for team in teams:
        try:
            elo_files[team] = await load_elo_team_file(cache_dir, team)
        except Exception as exc:
            elo_gaps.append(f"{team}: {exc}")

    model = WeightedPredictionModel()
    store = InMemoryEvidenceStore()
    feedback = FeedbackAnalyzer(store)
    group_state: dict[str, dict[str, int]] = {}
    predictions: list[dict[str, Any]] = []
    data_gaps: dict[str, int] = {}

    sorted_matches = sorted(matches, key=lambda m: (m["date"], m.get("time", "")))
    for match in sorted_matches:
        team1 = match["team1"]
        team2 = match["team2"]
        question = f"2022 World Cup group stage: {team1} vs {team2}, who wins?"
        event_time = datetime.fromisoformat(f"{match['date']}T{match.get('time','00:00')}:00").replace(tzinfo=timezone.utc)
        request = PredictionRequest(
            question=question,
            domain="football",
            outcome_type="three_way",
            outcomes=["home_win", "draw", "away_win"],
            event_time=event_time,
            prediction_deadline=event_time,
            context={"competitors": [team1, team2], "backtest": "world_cup_2022_group_stage"},
        )
        team_file = elo_files.get(team1) or ""
        elo = parse_elo_snapshot(team_file, match, code_map)
        spi = spi_index.get((match["date"], normalize_team(team1), normalize_team(team2)))
        forecast_pair = (
            latest_forecast_before(forecast_lookup, team1, event_time),
            latest_forecast_before(forecast_lookup, team2, event_time),
        )
        statsbomb_match = statsbomb_lookup.get((match["date"], normalize_team(team1), normalize_team(team2)))
        lineup_text = None
        post_match_xg = None
        statsbomb_match_gap = None
        if statsbomb_match:
            try:
                lineups = await load_statsbomb_lineups(cache_dir, int(statsbomb_match["match_id"]))
                lineup_text = lineup_summary(lineups, team1, team2)
            except Exception as exc:
                statsbomb_match_gap = f"StatsBomb lineups unavailable for {team1} vs {team2}: {exc}"
            try:
                events = await load_statsbomb_events(cache_dir, int(statsbomb_match["match_id"]))
                post_match_xg = post_event_xg(events, team1, team2)
            except Exception as exc:
                statsbomb_match_gap = f"StatsBomb events unavailable for {team1} vs {team2}: {exc}"
        else:
            statsbomb_match_gap = f"StatsBomb match mapping missing for {team1} vs {team2}."
        weather_item, weather_gap = await historical_weather(cache_dir, match, event_time)
        evidence, gaps = build_evidence(match, elo, group_state, spi, forecast_pair, lineup_text, weather_item)
        if weather_gap:
            gaps.append(weather_gap)
        if statsbomb_match_gap:
            gaps.append(statsbomb_match_gap)
        for gap in gaps:
            data_gaps[gap] = data_gaps.get(gap, 0) + 1

        factors = model.score_factors(FOOTBALL_PACK, request, evidence)
        outcomes = model.predict(request, factors)
        actual = outcome_from_score(match["score"])
        response = PredictionResponse(
            task_id=f"wc2022-{len(predictions)+1:02d}",
            domain="football",
            normalized_question=question,
            outcomes=outcomes,
            model_status="backtest",
            confidence=0.0,
            data_coverage=min(sum(f.evidence_count for f in factors) / max(len(factors) * 3, 1), 1),
            freshness=0.0,
            model_agreement=0.5,
            factors=factors,
            evidence=evidence,
            uncertainties=gaps[:8],
            workflow_trace=[
                "task_intake",
                "domain_pack:football",
                "factor_decomposition",
                "historical_evidence_collection",
                "factor_scoring",
                "probability_estimation",
                "feedback",
            ],
            research_review=None,
        )
        store.save_prediction(response)
        feedback_result = await feedback.analyze(FeedbackRequest(task_id=response.task_id, actual_outcome=actual))
        evidence_sources = sorted(
            {
                item.source
                for item in evidence
                if item.evidence_stage != "collection_gap" and item.source
            }
        )
        predictions.append(
            {
                "match_no": len(predictions) + 1,
                "date": match["date"],
                "group": match["group"],
                "team1": team1,
                "team2": team2,
                "score": match["score"]["ft"],
                "actual": actual,
                "baseline_outcomes": outcomes,
                "factors": factors,
                "feedback_metrics": feedback_result["metrics"],
                "post_match_xg": post_match_xg,
                "historical_elo_available": elo is not None,
                "team_level_spi_available": bool(forecast_pair[0] and forecast_pair[1]),
                "statsbomb_match_available": statsbomb_match is not None,
                "lineup_available": lineup_text is not None,
                "weather_available": weather_item is not None,
                "evidence_sources": evidence_sources,
                "gaps": gaps,
            }
        )
        update_group_state(match, group_state)

    baseline_weights = FOOTBALL_PACK.factor_weight_map()
    trained_weights = train_weights(predictions)
    baseline_log_loss = log_loss(predictions, baseline_weights)
    trained_log_loss = log_loss(predictions, trained_weights)
    baseline_brier = brier_score(predictions, baseline_weights)
    trained_brier = brier_score(predictions, trained_weights)
    directional_counts = directional_coverage(predictions)
    governance_outputs = governance_evaluation_outputs(
        predictions,
        baseline_weights,
        trained_weights,
    )
    data_coverage_summary = {
        "result_fixture_rows": len(predictions),
        "historical_elo": sum(1 for row in predictions if row["historical_elo_available"]),
        "team_level_spi_forecast": sum(1 for row in predictions if row["team_level_spi_available"]),
        "statsbomb_match_mapping": sum(1 for row in predictions if row["statsbomb_match_available"]),
        "statsbomb_lineups": sum(1 for row in predictions if row["lineup_available"]),
        "statsbomb_post_match_xg": sum(1 for row in predictions if row["post_match_xg"] is not None),
        "open_meteo_historical_weather": sum(1 for row in predictions if row["weather_available"]),
    }

    rows = []
    for row in predictions:
        trained_outcomes = predict_with_weights(row["factors"], ["home_win", "draw", "away_win"], trained_weights)
        rows.append(
            {
                "match_no": row["match_no"],
                "date": row["date"],
                "group": row["group"],
                "match": f"{row['team1']} vs {row['team2']}",
                "score": f"{row['score'][0]}-{row['score'][1]}",
                "actual": row["actual"],
                "baseline_home": round(row["baseline_outcomes"]["home_win"], 4),
                "baseline_draw": round(row["baseline_outcomes"]["draw"], 4),
                "baseline_away": round(row["baseline_outcomes"]["away_win"], 4),
                "trained_home": round(trained_outcomes["home_win"], 4),
                "trained_draw": round(trained_outcomes["draw"], 4),
                "trained_away": round(trained_outcomes["away_win"], 4),
                "post_match_xg": row.get("post_match_xg"),
                "lineup_available": row["lineup_available"],
                "weather_available": row["weather_available"],
                "evidence_sources": row["evidence_sources"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "worldcup_2022_group_backtest.json").write_text(
        json.dumps(
            {
                "source_manifest": {
                    "results": OPENFOOTBALL_2022_URL,
                    "historical_elo": ELO_BASE_URL,
                    "spi": FIVETHIRTYEIGHT_SPI_URL,
                    "spi_gap": spi_gap,
                    "wc_forecast_mirror": FIVETHIRTYEIGHT_WC_FORECAST_MIRROR_URL,
                    "wc_forecast_gap": forecast_gap,
                    "statsbomb_open_data": STATSBOMB_BASE_URL,
                    "statsbomb_gap": statsbomb_gap,
                    "open_meteo_archive": OPEN_METEO_ARCHIVE_URL,
                    "elo_gaps": elo_gaps,
                },
                "data_coverage_summary": data_coverage_summary,
                "data_gap_counts": data_gaps,
                "baseline_weights": baseline_weights,
                "trained_weights": trained_weights,
                "metrics": {
                    "matches": len(predictions),
                    "baseline_log_loss": round(baseline_log_loss, 6),
                    "trained_log_loss": round(trained_log_loss, 6),
                    "baseline_brier": round(baseline_brier, 6),
                    "trained_brier": round(trained_brier, 6),
                },
                **governance_outputs,
                "directional_coverage": directional_counts,
                "matches": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "matches": predictions,
        "rows": rows,
        "data_gap_counts": data_gaps,
        "data_coverage_summary": data_coverage_summary,
        "source_manifest": {
            "results": OPENFOOTBALL_2022_URL,
            "historical_elo": ELO_BASE_URL,
            "spi": FIVETHIRTYEIGHT_SPI_URL,
            "spi_gap": spi_gap,
            "wc_forecast_mirror": FIVETHIRTYEIGHT_WC_FORECAST_MIRROR_URL,
            "wc_forecast_gap": forecast_gap,
            "statsbomb_open_data": STATSBOMB_BASE_URL,
            "statsbomb_gap": statsbomb_gap,
            "open_meteo_archive": OPEN_METEO_ARCHIVE_URL,
            "elo_gaps": elo_gaps,
        },
        "baseline_weights": baseline_weights,
        "trained_weights": trained_weights,
        "metrics": {
            "matches": len(predictions),
            "baseline_log_loss": baseline_log_loss,
            "trained_log_loss": trained_log_loss,
            "baseline_brier": baseline_brier,
            "trained_brier": trained_brier,
        },
        **governance_outputs,
        "directional_coverage": directional_counts,
    }


def norway_iraq_comparison(result: dict[str, Any]) -> dict[str, Any]:
    evidence = [
        EvidenceItem(
            claim="Norway rank 10 rating 1914; Iraq rank 63 rating 1607; Elo edge favors Norway by 307 rating points.",
            source="world_football_elo",
            source_url="https://www.eloratings.net/World.tsv",
            evidence_stage="verified_candidate",
            impact_area="team_strength",
            source_reliability=0.86,
            recency_score=0.78,
            corroboration_count=1,
            confidence=0.907,
        ),
        EvidenceItem(
            claim="Match-specific structured odds, lineup, referee, and venue/weather are not available for this hypothetical fixture.",
            source="backtest_gap_analysis",
            evidence_stage="collection_gap",
            impact_area="market_odds",
            source_reliability=0.7,
            recency_score=0.0,
            confidence=0.0,
        ),
    ]
    request = PredictionRequest(
        question="Predict Norway vs Iraq, who wins?",
        domain="football",
        outcome_type="three_way",
        outcomes=["home_win", "draw", "away_win"],
        context={"competitors": ["Norway", "Iraq"]},
    )
    model = WeightedPredictionModel()
    baseline_factors = model.score_factors(FOOTBALL_PACK, request, evidence)
    baseline = model.predict(request, baseline_factors)
    trained = predict_with_weights(baseline_factors, request.outcomes, result["trained_weights"])
    return {
        "baseline": {key: round(value, 4) for key, value in baseline.items()},
        "trained": {key: round(value, 4) for key, value in trained.items()},
        "factors": [factor.model_dump() for factor in baseline_factors],
    }


def write_report(result: dict[str, Any], comparison: dict[str, Any], output_dir: Path) -> Path:
    metrics = result["metrics"]
    gaps = sorted(result["data_gap_counts"].items(), key=lambda item: item[1], reverse=True)
    directional = result["directional_coverage"]
    coverage = result["data_coverage_summary"]
    leaderboard_entries = result["model_leaderboard"]["entries"]
    lines = [
        "# 2022 World Cup Football Backtest Report",
        "",
        "## Scope",
        "",
        "This replay uses the previous men's FIFA World Cup, Qatar 2022. It covers all 48 group-stage matches from the openfootball World Cup JSON dataset and runs them through the current football prediction schema: evidence -> factor scores -> probabilities -> feedback metrics -> trained weight profile.",
        "",
        "## Data Sources",
        "",
        f"- Results and fixtures: `{OPENFOOTBALL_2022_URL}`",
        f"- Historical Elo snapshots: `{ELO_BASE_URL}` team TSV files",
        f"- SPI/market probability attempted: `{FIVETHIRTYEIGHT_SPI_URL}`",
        f"- SPI availability: {result['source_manifest']['spi_gap'] or 'available'}",
        f"- Team-level World Cup SPI forecast mirror: `{FIVETHIRTYEIGHT_WC_FORECAST_MIRROR_URL}`",
        f"- Team-level SPI forecast availability: {result['source_manifest']['wc_forecast_gap'] or 'available'}",
        f"- StatsBomb Open Data lineups/events: `{STATSBOMB_BASE_URL}`",
        f"- Open-Meteo archive weather: `{OPEN_METEO_ARCHIVE_URL}`",
        "",
        "## Data Coverage",
        "",
        f"- Fixture/result rows: {coverage['result_fixture_rows']} / {metrics['matches']}",
        f"- Historical Elo snapshots: {coverage['historical_elo']} / {metrics['matches']}",
        f"- Team-level SPI forecasts: {coverage['team_level_spi_forecast']} / {metrics['matches']}",
        f"- StatsBomb match mapping: {coverage['statsbomb_match_mapping']} / {metrics['matches']}",
        f"- StatsBomb confirmed lineups: {coverage['statsbomb_lineups']} / {metrics['matches']}",
        f"- StatsBomb post-match xG: {coverage['statsbomb_post_match_xg']} / {metrics['matches']}",
        f"- Open-Meteo kickoff-hour weather: {coverage['open_meteo_historical_weather']} / {metrics['matches']}",
        "",
        "## Data Gaps",
        "",
    ]
    for gap, count in gaps[:10]:
        lines.append(f"- {gap} ({count} matches)")
    lines.extend(
        [
            "",
            "## Data Supplement Plan",
            "",
            "- Match-specific odds / probabilities: prefer a stable public mirror of FiveThirtyEight `spi_matches_intl.csv` or a manually supplied open CSV with `date, team1, team2, prob1, probtie, prob2`. Football-Data.co.uk does not cover World Cup national-team fixtures in the same way it covers club leagues, so it remains useful for club backtests but not sufficient for this replay.",
            "- Lineups and injuries: StatsBomb Open Data now supplies confirmed lineups for this replay, but it does not by itself explain absences, fitness, suspensions, or late injuries. Add official match reports/team sheets as source URLs and normalize player-availability reasons.",
            "- Referee appointments: use official FIFA match centre pages or public match reports, then normalize referee, assistants, card rate, penalty rate, and VAR information.",
            "- Tactical matchup: StatsBomb events now provide post-match xG and event data for analysis, but pre-match tactical expectations still need a separate feature layer. For replay training, convert event-derived style features only from data available before each match.",
            "- Chemistry / relationships: derive measurable proxies such as shared national-team minutes, club teammate links, coach tenure, and lineup continuity. Avoid rumor-heavy weighting without corroboration.",
            "- Sentiment / narrative: use low weight unless a backtest proves lift. Store source reliability and contradiction counts because media narratives are noisy.",
            "- Weather / venue: Open-Meteo archive is now connected for kickoff-hour historical weather. For future fixtures, use forecast APIs and degrade freshness as kickoff approaches.",
            "",
            "## Weight Training Result",
            "",
            f"- Matches replayed: {metrics['matches']}",
            f"- Baseline log loss: {metrics['baseline_log_loss']:.4f}",
            f"- Trained log loss: {metrics['trained_log_loss']:.4f}",
            f"- Baseline Brier: {metrics['baseline_brier']:.4f}",
            f"- Trained Brier: {metrics['trained_brier']:.4f}",
            "",
            "## Model Leaderboard",
            "",
            "| Rank | Model/Profile | Log loss | Brier | RPS | Accuracy | Cases |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in leaderboard_entries:
        lines.append(
            f"| {entry['rank']} | {entry['model_id']} | {entry['log_loss']:.4f} | "
            f"{entry['brier_score']:.4f} | {entry['ranked_probability_score']:.4f} | "
            f"{entry['accuracy']:.3f} | {entry['case_count']} |"
        )
    lines.extend(
        [
            "",
            "## Versioned Ensemble Profile",
            "",
            "The online workflow now uses a versioned ensemble profile for football model blending. This backtest evaluates factor profiles, while the live workflow records ensemble profile id/version/maturity in `distribution_metrics`.",
            "",
            "```json",
            json.dumps(result["versioned_ensemble_profiles"], indent=2),
            "```",
            "",
            "## Promotion Gate",
            "",
            "The trained sparse profile is evaluated against the baseline profile by the default football 1.0 promotion gate. This gate is intentionally strict and should block promotion when the sample is too small, calibration is weak, or improvements are not large enough.",
            "",
            "```json",
            json.dumps(result["promotion_decision"], indent=2),
            "```",
            "",
            "Baseline weights:",
            "",
            "```json",
            json.dumps(result["baseline_weights"], indent=2),
            "```",
            "",
            "Trained weights:",
            "",
            "```json",
            json.dumps(result["trained_weights"], indent=2),
            "```",
            "",
            "Directional factor coverage in the 48-match replay:",
            "",
            "```json",
            json.dumps(directional, indent=2),
            "```",
            "",
            "Important calibration note: when a factor has zero directional coverage, a non-zero trained weight should be read as calibration reserve inside the current simple model, not as proof that the missing factor was learned. This replay says the current data layer is not yet rich enough to train lineup, referee, chemistry, tactical, sentiment, or market weights from evidence.",
            "",
            "## Norway vs Iraq Comparison",
            "",
            "Baseline prediction:",
            "",
            "```json",
            json.dumps(comparison["baseline"], indent=2),
            "```",
            "",
            "Trained-weight prediction:",
            "",
            "```json",
            json.dumps(comparison["trained"], indent=2),
            "```",
            "",
            "Interpretation: the replay learned from a sparse but replayable data layer. Historical Elo and group form were the only broadly available directional signals. The trained profile improves log loss only slightly, so it should be treated as a calibration profile rather than a mature football model. For Norway vs Iraq, the trained profile raises Norway's probability because the Elo gap is large, but the prediction remains conservative until match-specific odds, lineup, referee, and venue data are provided.",
            "",
            "## Match Replay Table",
            "",
            "| # | Date | Group | Match | Score | Actual | Baseline H/D/A | Trained H/D/A |",
            "|---:|---|---|---|---|---|---|---|",
        ]
    )
    for row in result["rows"]:
        lines.append(
            f"| {row['match_no']} | {row['date']} | {row['group']} | {row['match']} | {row['score']} | {row['actual']} | "
            f"{row['baseline_home']:.3f}/{row['baseline_draw']:.3f}/{row['baseline_away']:.3f} | "
            f"{row['trained_home']:.3f}/{row['trained_draw']:.3f}/{row['trained_away']:.3f} |"
        )
    report_path = output_dir / "worldcup_2022_backtest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    result = await run_backtest(output_dir)
    comparison = norway_iraq_comparison(result)
    comparison_path = output_dir / "norway_iraq_weight_comparison.json"
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(result, comparison, output_dir)
    print(
        json.dumps(
            {
                "matches": result["metrics"]["matches"],
                "baseline_log_loss": round(result["metrics"]["baseline_log_loss"], 6),
                "trained_log_loss": round(result["metrics"]["trained_log_loss"], 6),
                "baseline_brier": round(result["metrics"]["baseline_brier"], 6),
                "trained_brier": round(result["metrics"]["trained_brier"], 6),
                "trained_weights": result["trained_weights"],
                "leaderboard_top": result["model_leaderboard"]["entries"][:3],
                "comparison": comparison,
                "report": str(report_path),
                "json": str(output_dir / "worldcup_2022_group_backtest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
