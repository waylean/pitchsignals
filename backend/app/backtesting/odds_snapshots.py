from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

from app.backtesting.records import (
    DatasetManifest,
    FeatureSnapshot,
    FootballBacktestDataset,
    FootballMatchRecord,
)


HORIZON_HOURS = {
    "T-48h": 48,
    "T-24h": 24,
    "T-2h": 2,
    "T-1h": 1,
}

REQUIRED_ODDS_SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "match_id",
    "home_team",
    "away_team",
    "snapshot_at",
    "available_at",
    "home_odds",
    "draw_odds",
    "away_odds",
    "source_url",
    "license_note",
]
RECOMMENDED_ODDS_SNAPSHOT_COLUMNS = [
    "kickoff_at",
    "odds_type",
    "bookmaker",
    "book_count",
]


@dataclass(frozen=True)
class OddsSnapshot:
    snapshot_id: str
    match_id: str | None
    home_team: str
    away_team: str
    home_odds: float
    draw_odds: float
    away_odds: float
    snapshot_at: datetime
    available_at: datetime
    odds_type: str = "snapshot"
    bookmaker: str | None = None
    book_count: int = 1
    source_url: str | None = None
    source: str = "football_odds_snapshot_csv"
    raw: dict[str, Any] | None = None

    @property
    def overround(self) -> float:
        return round((1 / self.home_odds) + (1 / self.draw_odds) + (1 / self.away_odds) - 1, 6)

    @property
    def odds(self) -> dict[str, float]:
        return {
            "home": self.home_odds,
            "draw": self.draw_odds,
            "away": self.away_odds,
            "source": self.bookmaker or self.source,
            "overround": self.overround,
            "bookmaker_count": float(self.book_count),
        }


@dataclass(frozen=True)
class HorizonSelectionResult:
    records: list[FootballMatchRecord]
    selected_count: int
    missing_count: int
    leakage_issues: int
    horizon_counts: dict[str, int]
    gaps: list[str]


@dataclass(frozen=True)
class OddsSnapshotCsvAudit:
    source: str
    quality_gate: str
    release_gate: str
    total_rows: int
    valid_snapshots: int
    invalid_rows: int
    missing_required_columns: list[str]
    missing_recommended_columns: list[str]
    errors: list[str]
    warnings: list[str]
    duplicate_snapshot_ids: list[str]
    closing_snapshot_count: int
    date_only_timestamp_count: int
    coverage: dict[str, float]
    odds_type_counts: dict[str, int]


@dataclass(frozen=True)
class HorizonCoverageAudit:
    quality_gate: str
    release_gate: str
    base_records: int
    expected_horizon_records: int
    selected_horizon_records: int
    missing_horizon_records: int
    leakage_issues: int
    coverage_rate: float
    horizon_counts: dict[str, int]
    min_slice_coverage_rate: float
    slice_coverage: dict[str, dict[str, float | int | str]]
    failed_slices: list[dict[str, float | int | str]]
    errors: list[str]
    warnings: list[str]
    sample_gaps: list[str]


def load_odds_snapshot_csv_text(text: str, source: str = "football_odds_snapshot_csv") -> list[OddsSnapshot]:
    rows = list(csv.DictReader(StringIO(text.lstrip("\ufeff"))))
    return [snapshot for row in rows if (snapshot := odds_snapshot_from_row(row, source=source))]


def load_odds_snapshot_csv_path(path: Path) -> list[OddsSnapshot]:
    return load_odds_snapshot_csv_text(path.read_text(encoding="utf-8"), source=str(path))


def audit_odds_snapshot_csv_text(
    text: str,
    source: str = "football_odds_snapshot_csv",
) -> OddsSnapshotCsvAudit:
    reader = csv.DictReader(StringIO(text.lstrip("\ufeff")))
    rows = list(reader)
    fieldnames = list(reader.fieldnames or [])
    missing_required = [
        column for column in REQUIRED_ODDS_SNAPSHOT_COLUMNS if column not in fieldnames
    ]
    missing_recommended = [
        column for column in RECOMMENDED_ODDS_SNAPSHOT_COLUMNS if column not in fieldnames
    ]
    errors: list[str] = []
    warnings: list[str] = []
    if missing_required:
        errors.append(f"missing_required_columns: {', '.join(missing_required)}")
    if missing_recommended:
        warnings.append(f"missing_recommended_columns: {', '.join(missing_recommended)}")

    snapshots: list[OddsSnapshot] = []
    invalid_rows = 0
    snapshot_ids: dict[str, int] = {}
    duplicate_snapshot_ids: list[str] = []
    date_only_timestamp_count = 0
    odds_type_counts: dict[str, int] = {}
    for index, row in enumerate(rows, start=2):
        for key in ["snapshot_at", "available_at", "as_of"]:
            value = str(row.get(key) or "").strip()
            if value and _looks_date_only(value):
                date_only_timestamp_count += 1
        snapshot = odds_snapshot_from_row(row, source=source)
        if snapshot is None:
            invalid_rows += 1
            errors.append(f"row {index}: invalid or incomplete odds snapshot")
            continue
        snapshots.append(snapshot)
        snapshot_ids[snapshot.snapshot_id] = snapshot_ids.get(snapshot.snapshot_id, 0) + 1
        if snapshot_ids[snapshot.snapshot_id] == 2:
            duplicate_snapshot_ids.append(snapshot.snapshot_id)
        odds_type_counts[snapshot.odds_type] = odds_type_counts.get(snapshot.odds_type, 0) + 1

    if duplicate_snapshot_ids:
        errors.append(f"duplicate_snapshot_ids: {', '.join(sorted(duplicate_snapshot_ids)[:10])}")
    if date_only_timestamp_count:
        warnings.append(f"date_only_timestamps: {date_only_timestamp_count}")
    closing_snapshot_count = sum(
        1 for snapshot in snapshots if snapshot.odds_type.lower() in {"closing", "close", "settled"}
    )
    if closing_snapshot_count:
        warnings.append(f"closing_snapshots_present: {closing_snapshot_count}")
    if not snapshots:
        errors.append("no_valid_snapshots")

    coverage = _snapshot_field_coverage(rows)
    for field in ["source_url", "license_note", "match_id", "available_at", "snapshot_at"]:
        if coverage.get(field, 0.0) < 1.0:
            warnings.append(f"{field}_coverage {coverage.get(field, 0.0):.3f} < 1.000")

    quality_gate = "pass"
    if errors:
        quality_gate = "fail"
    elif warnings:
        quality_gate = "warn"
    release_gate = "blocks_1_0" if errors else ("allows_prerelease" if warnings else "allows_1_0")
    return OddsSnapshotCsvAudit(
        source=source,
        quality_gate=quality_gate,
        release_gate=release_gate,
        total_rows=len(rows),
        valid_snapshots=len(snapshots),
        invalid_rows=invalid_rows,
        missing_required_columns=missing_required,
        missing_recommended_columns=missing_recommended,
        errors=errors,
        warnings=warnings,
        duplicate_snapshot_ids=sorted(duplicate_snapshot_ids),
        closing_snapshot_count=closing_snapshot_count,
        date_only_timestamp_count=date_only_timestamp_count,
        coverage=coverage,
        odds_type_counts=odds_type_counts,
    )


def audit_odds_snapshot_csv_path(path: Path) -> OddsSnapshotCsvAudit:
    return audit_odds_snapshot_csv_text(path.read_text(encoding="utf-8"), source=str(path))


def audit_horizon_coverage(
    base: FootballBacktestDataset,
    snapshots: Iterable[OddsSnapshot],
    horizons: Iterable[str] = ("T-48h", "T-24h", "T-2h", "T-1h"),
    min_coverage_rate: float = 0.95,
    min_slice_coverage_rate: float | None = None,
) -> HorizonCoverageAudit:
    horizon_list = list(horizons)
    slice_threshold = min_coverage_rate if min_slice_coverage_rate is None else min_slice_coverage_rate
    expected = len(base.records) * len(horizon_list)
    selection = materialize_horizon_records(base.records, snapshots, horizons=horizon_list)
    coverage_rate = selection.selected_count / expected if expected else 0.0
    slice_coverage, failed_slices = _horizon_slice_coverage(
        base.records,
        selection.records,
        horizon_list,
        min_slice_coverage_rate=slice_threshold,
    )
    errors: list[str] = []
    warnings: list[str] = []
    if selection.leakage_issues:
        errors.append(f"leakage_issues: {selection.leakage_issues}")
    if coverage_rate < min_coverage_rate:
        errors.append(
            f"horizon_coverage_rate {coverage_rate:.3f} < required {min_coverage_rate:.3f}"
        )
    if failed_slices:
        preview = ", ".join(str(item["slice_id"]) for item in failed_slices[:10])
        errors.append(
            f"horizon_slice_coverage_failed {len(failed_slices)} slices below "
            f"{slice_threshold:.3f}: {preview}"
        )
    for horizon in horizon_list:
        if selection.horizon_counts.get(horizon, 0) == 0:
            warnings.append(f"{horizon}: no selected records")
    quality_gate = "fail" if errors else ("warn" if warnings else "pass")
    release_gate = "blocks_1_0" if errors else ("allows_prerelease" if warnings else "allows_1_0")
    return HorizonCoverageAudit(
        quality_gate=quality_gate,
        release_gate=release_gate,
        base_records=len(base.records),
        expected_horizon_records=expected,
        selected_horizon_records=selection.selected_count,
        missing_horizon_records=selection.missing_count,
        leakage_issues=selection.leakage_issues,
        coverage_rate=round(coverage_rate, 6),
        horizon_counts=selection.horizon_counts,
        min_slice_coverage_rate=round(slice_threshold, 6),
        slice_coverage=slice_coverage,
        failed_slices=failed_slices[:100],
        errors=errors,
        warnings=warnings,
        sample_gaps=selection.gaps[:20],
    )


def _horizon_slice_coverage(
    base_records: Iterable[FootballMatchRecord],
    selected_records: Iterable[FootballMatchRecord],
    horizons: Iterable[str],
    min_slice_coverage_rate: float,
) -> tuple[dict[str, dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    expected_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    horizon_list = list(horizons)
    for record in base_records:
        for horizon in horizon_list:
            key = _horizon_slice_key(record.competition, record.season, horizon)
            expected_counts[key] = expected_counts.get(key, 0) + 1
    for record in selected_records:
        key = _horizon_slice_key(record.competition, record.season, record.odds_snapshot_type)
        selected_counts[key] = selected_counts.get(key, 0) + 1

    coverage: dict[str, dict[str, float | int | str]] = {}
    failed: list[dict[str, float | int | str]] = []
    for key in sorted(expected_counts):
        expected = expected_counts[key]
        selected = selected_counts.get(key, 0)
        rate = selected / expected if expected else 0.0
        item: dict[str, float | int | str] = {
            "slice_id": key,
            "expected": expected,
            "selected": selected,
            "missing": expected - selected,
            "coverage_rate": round(rate, 6),
        }
        coverage[key] = item
        if rate < min_slice_coverage_rate:
            failed.append(item)
    return coverage, failed


def _horizon_slice_key(competition: str, season: str, horizon: str) -> str:
    return f"{competition}:{season}:{horizon}"


def odds_snapshot_from_row(row: dict[str, Any], source: str = "football_odds_snapshot_csv") -> OddsSnapshot | None:
    try:
        home_odds = float(row.get("home_odds") or row.get("home") or row.get("h"))
        draw_odds = float(row.get("draw_odds") or row.get("draw") or row.get("d"))
        away_odds = float(row.get("away_odds") or row.get("away") or row.get("a"))
    except (TypeError, ValueError):
        return None
    if min(home_odds, draw_odds, away_odds) <= 1:
        return None

    snapshot_at = _parse_datetime(row.get("snapshot_at") or row.get("as_of"))
    available_at = _parse_datetime(row.get("available_at") or row.get("as_of") or row.get("snapshot_at"))
    home_team = str(row.get("home_team") or row.get("team1") or "").strip()
    away_team = str(row.get("away_team") or row.get("team2") or "").strip()
    if not snapshot_at or not available_at or not home_team or not away_team:
        return None

    snapshot_id = str(
        row.get("snapshot_id")
        or row.get("id")
        or ":".join(
            [
                "odds",
                _normalize(home_team),
                _normalize(away_team),
                snapshot_at.isoformat(),
                str(row.get("bookmaker") or row.get("source") or "snapshot"),
            ]
        )
    )
    try:
        book_count = int(float(row.get("book_count") or row.get("bookmaker_count") or 1))
    except (TypeError, ValueError):
        book_count = 1
    return OddsSnapshot(
        snapshot_id=snapshot_id,
        match_id=_optional_str(row.get("match_id")),
        home_team=home_team,
        away_team=away_team,
        home_odds=home_odds,
        draw_odds=draw_odds,
        away_odds=away_odds,
        snapshot_at=snapshot_at,
        available_at=available_at,
        odds_type=str(row.get("odds_type") or row.get("snapshot_type") or "snapshot"),
        bookmaker=_optional_str(row.get("bookmaker") or row.get("source")),
        book_count=max(book_count, 1),
        source_url=_optional_str(row.get("source_url")),
        source=source,
        raw=dict(row),
    )


def select_odds_snapshot_for_horizon(
    record: FootballMatchRecord,
    snapshots: Iterable[OddsSnapshot],
    horizon_profile: str,
) -> OddsSnapshot | None:
    deadline = prediction_deadline_for_horizon(record.event_time, horizon_profile)
    candidates = [
        snapshot
        for snapshot in snapshots
        if _matches_record(record, snapshot)
        and snapshot.available_at <= deadline
        and snapshot.snapshot_at <= deadline
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda snapshot: (snapshot.available_at, snapshot.snapshot_at))[-1]


def materialize_horizon_records(
    records: Iterable[FootballMatchRecord],
    snapshots: Iterable[OddsSnapshot],
    horizons: Iterable[str] = ("T-48h", "T-24h", "T-2h", "T-1h"),
) -> HorizonSelectionResult:
    snapshot_list = list(snapshots)
    materialized: list[FootballMatchRecord] = []
    gaps: list[str] = []
    horizon_counts: dict[str, int] = {horizon: 0 for horizon in horizons}
    for record in records:
        for horizon in horizons:
            selected = select_odds_snapshot_for_horizon(record, snapshot_list, horizon)
            if selected is None:
                gaps.append(f"{record.match_id}:{horizon}:missing legal odds snapshot")
                continue
            deadline = prediction_deadline_for_horizon(record.event_time, horizon)
            odds_snapshot = FeatureSnapshot(
                snapshot_id=selected.snapshot_id,
                source=selected.source,
                snapshot_type=horizon,
                available_at=selected.available_at,
                confidence=0.9,
                values={
                    "home": selected.home_odds,
                    "draw": selected.draw_odds,
                    "away": selected.away_odds,
                    "overround": selected.overround,
                    "bookmaker": selected.bookmaker,
                    "bookmaker_count": selected.book_count,
                    "odds_type": selected.odds_type,
                    "snapshot_at": selected.snapshot_at.isoformat(),
                    "source_url": selected.source_url,
                },
            )
            materialized.append(
                replace(
                    record,
                    match_id=f"{record.match_id}:{horizon}",
                    odds=selected.odds,
                    prediction_deadline=deadline,
                    odds_as_of=selected.snapshot_at,
                    odds_available_at=selected.available_at,
                    feature_available_at=selected.available_at,
                    source_snapshot_id=selected.snapshot_id,
                    odds_snapshot_type=horizon,
                    odds_snapshot=odds_snapshot,
                )
            )
            horizon_counts[horizon] = horizon_counts.get(horizon, 0) + 1
    leakage_issues = sum(len(record.leakage_issues()) for record in materialized)
    return HorizonSelectionResult(
        records=materialized,
        selected_count=len(materialized),
        missing_count=len(gaps),
        leakage_issues=leakage_issues,
        horizon_counts=horizon_counts,
        gaps=gaps,
    )


def build_horizon_odds_dataset(
    base: FootballBacktestDataset,
    snapshots: Iterable[OddsSnapshot],
    horizons: Iterable[str] = ("T-48h", "T-24h", "T-2h", "T-1h"),
    dataset_id: str | None = None,
) -> FootballBacktestDataset:
    selection = materialize_horizon_records(base.records, snapshots, horizons=horizons)
    manifest = DatasetManifest(
        dataset_id=dataset_id or f"{base.manifest.dataset_id}_horizon_odds",
        source_url=base.manifest.source_url,
        license_note=base.manifest.license_note,
        cache_path=base.manifest.cache_path,
        competition=base.manifest.competition,
        season=base.manifest.season,
        case_count=len(selection.records),
        dataset_hash=base.manifest.dataset_hash,
        downloaded_at=base.manifest.downloaded_at,
        snapshot_policy="odds_snapshot_csv_available_at",
        horizon_policy=",".join(horizons),
        license_tier=base.manifest.license_tier,
        redistribution_allowed=base.manifest.redistribution_allowed,
        coverage={
            **base.manifest.coverage,
            "horizon_records": len(selection.records),
            "horizon_missing_snapshots": selection.missing_count,
            "horizon_leakage_issues": selection.leakage_issues,
            **{f"horizon_{key}_records": value for key, value in selection.horizon_counts.items()},
        },
        gaps=[*base.manifest.gaps, *selection.gaps[:20]],
    )
    return FootballBacktestDataset(manifest=manifest, records=selection.records)


def prediction_deadline_for_horizon(event_time: datetime, horizon_profile: str) -> datetime:
    try:
        hours = HORIZON_HOURS[horizon_profile]
    except KeyError as exc:
        raise ValueError(f"Unsupported horizon profile: {horizon_profile}") from exc
    return event_time - timedelta(hours=hours)


def _matches_record(record: FootballMatchRecord, snapshot: OddsSnapshot) -> bool:
    if snapshot.match_id:
        return snapshot.match_id == record.match_id
    return (
        _normalize(record.home_team) == _normalize(snapshot.home_team)
        and _normalize(record.away_team) == _normalize(snapshot.away_team)
    )


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%d/%m/%Y"]:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _snapshot_field_coverage(rows: list[dict[str, Any]]) -> dict[str, float]:
    fields = [
        "snapshot_id",
        "match_id",
        "home_team",
        "away_team",
        "kickoff_at",
        "snapshot_at",
        "available_at",
        "odds_type",
        "bookmaker",
        "book_count",
        "source_url",
        "license_note",
    ]
    if not rows:
        return {field: 0.0 for field in fields}
    return {
        field: round(
            sum(1 for row in rows if str(row.get(field) or "").strip()) / len(rows),
            6,
        )
        for field in fields
    }


def _looks_date_only(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    return len(text) <= 10 and "T" not in text and ":" not in text


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("&", "and").split())


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
