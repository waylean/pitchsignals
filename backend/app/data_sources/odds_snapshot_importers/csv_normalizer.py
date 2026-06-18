from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha1
from io import StringIO
from pathlib import Path
from typing import Any

from app.backtesting.odds_snapshots import (
    RECOMMENDED_ODDS_SNAPSHOT_COLUMNS,
    REQUIRED_ODDS_SNAPSHOT_COLUMNS,
    OddsSnapshotCsvAudit,
    audit_odds_snapshot_csv_text,
)


CANONICAL_ODDS_SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "match_id",
    "home_team",
    "away_team",
    "kickoff_at",
    "snapshot_at",
    "available_at",
    "odds_type",
    "bookmaker",
    "home_odds",
    "draw_odds",
    "away_odds",
    "book_count",
    "source_url",
    "license_note",
]


DEFAULT_ALIASES: dict[str, list[str]] = {
    "snapshot_id": ["snapshot_id", "id", "odds_snapshot_id"],
    "match_id": ["match_id", "fixture_id", "game_id"],
    "home_team": ["home_team", "HomeTeam", "home", "team1", "Home"],
    "away_team": ["away_team", "AwayTeam", "away", "team2", "Away"],
    "kickoff_at": ["kickoff_at", "kickoff", "event_time", "match_time", "Date", "date"],
    "snapshot_at": ["snapshot_at", "as_of", "timestamp", "odds_timestamp", "collected_at"],
    "available_at": ["available_at", "published_at", "as_of", "timestamp", "odds_timestamp"],
    "odds_type": ["odds_type", "snapshot_type", "market_time"],
    "bookmaker": ["bookmaker", "book", "source", "odds_source"],
    "home_odds": ["home_odds", "home_price", "B365H", "AvgH", "MaxH", "PSH", "home"],
    "draw_odds": ["draw_odds", "draw_price", "B365D", "AvgD", "MaxD", "PSD", "draw"],
    "away_odds": ["away_odds", "away_price", "B365A", "AvgA", "MaxA", "PSA", "away"],
    "book_count": ["book_count", "bookmaker_count", "bookmakers"],
    "source_url": ["source_url", "url", "source_link"],
    "license_note": ["license_note", "license", "terms_note"],
}


@dataclass(frozen=True)
class NormalizedOddsSnapshotCsv:
    text: str
    audit: OddsSnapshotCsvAudit
    row_count: int


def normalize_odds_snapshot_csv_path(
    input_path: Path,
    *,
    source_url_default: str,
    license_note: str,
    output_path: Path | None = None,
    mapping: dict[str, str] | None = None,
    bookmaker_default: str = "unknown",
    odds_type_default: str = "current",
) -> NormalizedOddsSnapshotCsv:
    result = normalize_odds_snapshot_csv_text(
        input_path.read_text(encoding="utf-8"),
        source_url_default=source_url_default,
        license_note=license_note,
        mapping=mapping,
        bookmaker_default=bookmaker_default,
        odds_type_default=odds_type_default,
        source=str(input_path),
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.text, encoding="utf-8")
    return result


def normalize_odds_snapshot_csv_text(
    text: str,
    *,
    source_url_default: str,
    license_note: str,
    mapping: dict[str, str] | None = None,
    bookmaker_default: str = "unknown",
    odds_type_default: str = "current",
    source: str = "normalized_odds_snapshot_csv",
) -> NormalizedOddsSnapshotCsv:
    rows = list(csv.DictReader(StringIO(text.lstrip("\ufeff"))))
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=CANONICAL_ODDS_SNAPSHOT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = _normalize_row(
            row,
            mapping=mapping or {},
            source_url_default=source_url_default,
            license_note=license_note,
            bookmaker_default=bookmaker_default,
            odds_type_default=odds_type_default,
        )
        writer.writerow(normalized)
    normalized_text = output.getvalue()
    audit = audit_odds_snapshot_csv_text(normalized_text, source=source)
    return NormalizedOddsSnapshotCsv(
        text=normalized_text,
        audit=audit,
        row_count=len(rows),
    )


def _normalize_row(
    row: dict[str, Any],
    *,
    mapping: dict[str, str],
    source_url_default: str,
    license_note: str,
    bookmaker_default: str,
    odds_type_default: str,
) -> dict[str, str]:
    values = {
        column: _value(row, column, mapping)
        for column in CANONICAL_ODDS_SNAPSHOT_COLUMNS
    }
    values["source_url"] = values["source_url"] or source_url_default
    values["license_note"] = values["license_note"] or license_note
    values["bookmaker"] = values["bookmaker"] or bookmaker_default
    values["odds_type"] = values["odds_type"] or odds_type_default
    values["available_at"] = values["available_at"] or values["snapshot_at"]
    values["book_count"] = values["book_count"] or "1"
    if not values["match_id"]:
        values["match_id"] = _stable_id(
            "match",
            values["home_team"],
            values["away_team"],
            values["kickoff_at"],
        )
    if not values["snapshot_id"]:
        values["snapshot_id"] = _stable_id(
            "odds",
            values["match_id"],
            values["snapshot_at"],
            values["bookmaker"],
            values["odds_type"],
        )
    return {column: values.get(column, "") for column in CANONICAL_ODDS_SNAPSHOT_COLUMNS}


def _value(row: dict[str, Any], column: str, mapping: dict[str, str]) -> str:
    explicit = mapping.get(column)
    if explicit:
        return _clean(row.get(explicit))
    for alias in DEFAULT_ALIASES.get(column, []):
        value = _clean(row.get(alias))
        if value:
            return value
    return ""


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "|".join(_normalize(part) for part in parts if part)
    digest = sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("&", "and").split())


def required_and_recommended_columns() -> list[str]:
    return list(dict.fromkeys([*REQUIRED_ODDS_SNAPSHOT_COLUMNS, *RECOMMENDED_ODDS_SNAPSHOT_COLUMNS]))
