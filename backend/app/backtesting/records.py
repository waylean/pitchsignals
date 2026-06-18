from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    source_url: str
    license_note: str
    cache_path: str | None = None
    competition: str | None = None
    season: str | None = None
    case_count: int = 0
    dataset_hash: str | None = None
    downloaded_at: str | None = None
    snapshot_policy: str | None = None
    horizon_policy: str | None = None
    license_tier: str = "public_free"
    redistribution_allowed: bool = False
    coverage: dict[str, int | float | str | bool | None] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FeatureSnapshot:
    snapshot_id: str
    source: str
    snapshot_type: str
    available_at: datetime | None = None
    confidence: float | None = None
    values: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FootballMatchRecord:
    match_id: str
    dataset_id: str
    competition: str
    season: str
    event_time: datetime
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    neutral_site: bool = False
    odds: dict[str, float] = field(default_factory=dict)
    prediction_deadline: datetime | None = None
    odds_as_of: datetime | None = None
    odds_available_at: datetime | None = None
    feature_available_at: datetime | None = None
    source_snapshot_id: str | None = None
    odds_snapshot_type: str = "unknown"
    odds_snapshot: FeatureSnapshot | None = None
    lineup_snapshot: FeatureSnapshot | None = None
    weather_snapshot: FeatureSnapshot | None = None
    referee_snapshot: FeatureSnapshot | None = None
    feature_snapshots: list[FeatureSnapshot] = field(default_factory=list)
    source_row: dict[str, Any] = field(default_factory=dict)

    @property
    def actual_result(self) -> str:
        if self.home_goals > self.away_goals:
            return "home_win"
        if self.away_goals > self.home_goals:
            return "away_win"
        return "draw"

    @property
    def has_1x2_odds(self) -> bool:
        return all(self.odds.get(key, 0) > 1 for key in ["home", "draw", "away"])

    def leakage_issues(self) -> list[str]:
        issues: list[str] = []
        deadline = self.prediction_deadline or self.event_time
        snapshots = [
            snapshot
            for snapshot in [
                self.odds_snapshot,
                self.lineup_snapshot,
                self.weather_snapshot,
                self.referee_snapshot,
                *self.feature_snapshots,
            ]
            if snapshot is not None
        ]
        for snapshot in snapshots:
            if snapshot.available_at and snapshot.available_at > deadline:
                issues.append(
                    f"{snapshot.snapshot_type}:{snapshot.snapshot_id} available_at "
                    f"{snapshot.available_at.isoformat()} after prediction_deadline {deadline.isoformat()}"
                )
            snapshot_at = snapshot.values.get("snapshot_at")
            if isinstance(snapshot_at, str):
                try:
                    parsed_snapshot_at = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
                except ValueError:
                    parsed_snapshot_at = None
                if parsed_snapshot_at and parsed_snapshot_at > deadline:
                    issues.append(
                        f"{snapshot.snapshot_type}:{snapshot.snapshot_id} snapshot_at "
                        f"{parsed_snapshot_at.isoformat()} after prediction_deadline {deadline.isoformat()}"
                    )
        return issues


@dataclass(frozen=True)
class FootballBacktestDataset:
    manifest: DatasetManifest
    records: list[FootballMatchRecord]
