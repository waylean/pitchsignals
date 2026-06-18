from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FootballDatasetSpec:
    dataset_id: str
    label: str
    source_url: str
    access: str
    license_note: str
    competitions: list[str]
    fields: list[str]
    feature_targets: list[str]
    backtest_use: list[str]
    priority: int
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "label": self.label,
            "source_url": self.source_url,
            "access": self.access,
            "license_note": self.license_note,
            "competitions": self.competitions,
            "fields": self.fields,
            "feature_targets": self.feature_targets,
            "backtest_use": self.backtest_use,
            "priority": self.priority,
            "risks": self.risks,
            "notes": self.notes,
        }


FOOTBALL_1_0_DATASETS: list[FootballDatasetSpec] = [
    FootballDatasetSpec(
        dataset_id="football_data_co_uk",
        label="Football-Data.co.uk club results and odds",
        source_url="https://www.football-data.co.uk/data.php",
        access="public_free_csv",
        license_note="Public CSV downloads; keep source attribution and do not redistribute as proprietary data.",
        competitions=[
            "Premier League",
            "Championship",
            "La Liga",
            "Bundesliga",
            "Serie A",
            "Ligue 1",
            "major European club leagues",
        ],
        fields=[
            "date",
            "home_team",
            "away_team",
            "full_time_score",
            "full_time_result",
            "1x2 bookmaker odds",
            "average odds",
            "maximum odds",
            "opening/closing columns where available",
            "totals and Asian handicap columns where available",
        ],
        feature_targets=["market_odds", "team_strength", "xg_form"],
        backtest_use=[
            "market-only baseline",
            "club-league time-split backtest",
            "CLV/open-current-closing research where columns are available",
        ],
        priority=1,
        risks=[
            "Primarily club football, not World Cup national-team fixtures.",
            "Column availability varies by season and league.",
        ],
    ),
    FootballDatasetSpec(
        dataset_id="world_football_elo",
        label="World Football Elo Ratings",
        source_url="https://www.eloratings.net/",
        access="public_free_tsv",
        license_note="Public ratings pages/TSV; store snapshots and source URLs.",
        competitions=["national teams", "World Cup", "European Championship", "continental cups"],
        fields=["team", "rating", "rank", "historical match Elo rows", "result history"],
        feature_targets=["team_strength", "xg_form"],
        backtest_use=["national-team baseline", "strength prior", "rating-difference calibration"],
        priority=1,
        risks=["Country naming must be normalized carefully."],
    ),
    FootballDatasetSpec(
        dataset_id="openfootball_worldcup",
        label="openfootball World Cup JSON",
        source_url="https://github.com/openfootball/worldcup.json",
        access="public_open_data",
        license_note="Open public repository; follow repository license and attribution.",
        competitions=["World Cup historical tournaments"],
        fields=["fixtures", "groups", "teams", "scores", "dates", "venues where available"],
        feature_targets=["team_strength", "environment"],
        backtest_use=["World Cup replay", "fixture/result ground truth", "tournament-state features"],
        priority=1,
        risks=["Limited odds, lineup, and referee fields."],
    ),
    FootballDatasetSpec(
        dataset_id="openfootball_euro",
        label="openfootball Euro JSON",
        source_url="https://github.com/openfootball/euro.json",
        access="public_open_data",
        license_note="Open public repository; follow repository license and attribution.",
        competitions=["UEFA European Championship historical tournaments"],
        fields=["fixtures", "groups", "teams", "scores", "dates", "venues where available"],
        feature_targets=["team_strength", "environment"],
        backtest_use=["Euro replay", "national-team validation beyond World Cup"],
        priority=1,
        risks=["Limited market and lineup data."],
    ),
    FootballDatasetSpec(
        dataset_id="openfootball_club_leagues",
        label="openfootball football.json club league data",
        source_url="https://github.com/openfootball/football.json",
        access="public_open_data",
        license_note="Open public repository; follow repository license and attribution.",
        competitions=["Premier League", "Bundesliga", "La Liga", "Serie A", "Ligue 1"],
        fields=["fixtures", "results", "teams", "rounds", "dates"],
        feature_targets=["team_strength", "environment"],
        backtest_use=["club fixture/result validation", "schedule density and form features"],
        priority=1,
        risks=["Limited market, lineup, and referee data."],
    ),
    FootballDatasetSpec(
        dataset_id="statsbomb_open_data",
        label="StatsBomb Open Data",
        source_url="https://github.com/statsbomb/open-data",
        access="public_open_data",
        license_note="Open data with StatsBomb attribution requirements; respect repository terms.",
        competitions=["selected World Cup", "selected club and international competitions"],
        fields=[
            "events",
            "shots",
            "xg",
            "lineups",
            "player ids",
            "positions",
            "set pieces",
            "passes",
            "pressures where available",
        ],
        feature_targets=["xg_form", "lineup_availability", "tactical_matchup"],
        backtest_use=[
            "post-match xG training labels",
            "style features from matches before prediction deadline",
            "lineup continuity and role-availability proxies",
        ],
        priority=2,
        risks=[
            "Coverage is selective, not universal.",
            "Post-match events must never leak into same-match prematch predictions.",
        ],
    ),
    FootballDatasetSpec(
        dataset_id="open_meteo",
        label="Open-Meteo forecast and historical weather",
        source_url="https://open-meteo.com/",
        access="public_free_api",
        license_note="Free public API; cite source and cache responsibly.",
        competitions=["all competitions with venue coordinates"],
        fields=["temperature", "precipitation", "wind", "humidity", "weather_code", "forecast/historical timestamp"],
        feature_targets=["referee_environment", "environment"],
        backtest_use=["weather/environment features by horizon"],
        priority=2,
        risks=["Historical archive is not the same as historical forecast snapshot."],
    ),
    FootballDatasetSpec(
        dataset_id="clubelo",
        label="ClubElo ratings",
        source_url="https://clubelo.com/",
        access="public_web_csv",
        license_note="Public rating data; license must be reviewed before caching or redistribution.",
        competitions=["European club leagues", "continental club competitions"],
        fields=["club", "rating", "date", "country", "coach fields where available"],
        feature_targets=["team_strength", "tactical_matchup"],
        backtest_use=["club strength prior", "club-league Elo baseline"],
        priority=2,
        risks=["License clarity is weaker than CC0/openfootball sources."],
    ),
    FootballDatasetSpec(
        dataset_id="wikidata_reep_entities",
        label="Wikidata / REEP football entity alignment",
        source_url="https://www.wikidata.org/",
        access="public_open_data",
        license_note="Wikidata is CC0; REEP or other mirrors require repository-specific review.",
        competitions=["all competitions"],
        fields=["team ids", "player ids", "coach ids", "nationality", "club affiliation", "entity aliases"],
        feature_targets=["lineup_availability", "team_strength"],
        backtest_use=["entity normalization", "player/team alias matching for public data joins"],
        priority=2,
        risks=["Entity joins can introduce false matches without careful normalization."],
    ),
    FootballDatasetSpec(
        dataset_id="official_fixture_pages",
        label="FIFA/UEFA official fixtures and match centres",
        source_url="https://www.fifa.com/ and https://www.uefa.com/",
        access="public_web_pages",
        license_note="Use as source URLs and extracted factual metadata; do not bulk republish copyrighted pages.",
        competitions=["World Cup", "Euro", "Champions League", "Europa League"],
        fields=["kickoff", "venue", "teams", "official result", "referee where published", "lineups where published"],
        feature_targets=["referee_environment", "lineup_availability", "environment"],
        backtest_use=["official validation and metadata cross-check"],
        priority=2,
        risks=["Page structure changes; scraping rules must be respected."],
    ),
    FootballDatasetSpec(
        dataset_id="kaggle_international_results_optional",
        label="International football results mirror",
        source_url="https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017",
        access="optional_manual_download",
        license_note="Do not make Kaggle a default runtime dependency; user must review dataset license.",
        competitions=["national-team friendlies and tournaments"],
        fields=["date", "home_team", "away_team", "score", "tournament", "city", "country", "neutral"],
        feature_targets=["team_strength", "environment"],
        backtest_use=["long-history national-team model validation"],
        priority=3,
        risks=["Manual download and license review required before redistribution."],
    ),
    FootballDatasetSpec(
        dataset_id="understat_optional",
        label="Understat optional xG pages",
        source_url="https://understat.com/",
        access="optional_user_enabled_web",
        license_note="Free web pages but redistribution/scraping terms are not clear; not a default open-source dependency.",
        competitions=["top European club leagues"],
        fields=["team xG", "player xG", "shots", "xGA"],
        feature_targets=["xg_form", "tactical_matchup"],
        backtest_use=["optional club xG validation when user explicitly enables it"],
        priority=4,
        risks=["ToS/licensing uncertainty; keep optional and do not redistribute raw data."],
    ),
    FootballDatasetSpec(
        dataset_id="restricted_sources_not_default",
        label="Restricted or high-risk football sources",
        source_url="FBref / Transfermarkt / OddsPortal / paid news and social sources",
        access="not_default",
        license_note="Do not make these core dependencies; only support user-provided local snapshots when legally obtained.",
        competitions=["various"],
        fields=["injuries", "market values", "odds history", "advanced stats", "news narratives"],
        feature_targets=[
            "market_odds",
            "lineup_availability",
            "xg_form",
            "sentiment_narrative",
        ],
        backtest_use=["local user experiments only"],
        priority=5,
        risks=[
            "Terms of service, copyright, login/CAPTCHA, and redistribution risks.",
            "Not suitable as an open-source default data path.",
        ],
    ),
]


def football_1_0_dataset_matrix() -> list[dict[str, Any]]:
    return [dataset.to_dict() for dataset in sorted(FOOTBALL_1_0_DATASETS, key=lambda item: item.priority)]


def required_feature_coverage() -> dict[str, list[str]]:
    coverage: dict[str, list[str]] = {}
    for dataset in FOOTBALL_1_0_DATASETS:
        for target in dataset.feature_targets:
            coverage.setdefault(target, []).append(dataset.dataset_id)
    return coverage
