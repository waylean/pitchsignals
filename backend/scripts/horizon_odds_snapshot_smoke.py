from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.backtesting.odds_snapshots import (  # noqa: E402
    audit_horizon_coverage,
    audit_odds_snapshot_csv_text,
    build_horizon_odds_dataset,
    load_odds_snapshot_csv_text,
    materialize_horizon_records,
    prediction_deadline_for_horizon,
    select_odds_snapshot_for_horizon,
)
from app.backtesting.records import DatasetManifest, FootballBacktestDataset, FootballMatchRecord  # noqa: E402


def main() -> None:
    kickoff = datetime.fromisoformat("2026-06-20T20:00:00")
    record = FootballMatchRecord(
        match_id="fixture-001",
        dataset_id="fixture",
        competition="fixture_cup",
        season="2026",
        event_time=kickoff,
        home_team="Norway",
        away_team="Iraq",
        home_goals=2,
        away_goals=0,
        odds={"home": 1.50, "draw": 4.20, "away": 7.00},
    )
    rematch = FootballMatchRecord(
        match_id="fixture-002",
        dataset_id="fixture",
        competition="fixture_cup",
        season="2026",
        event_time=kickoff,
        home_team="Norway",
        away_team="Iraq",
        home_goals=1,
        away_goals=1,
        odds={"home": 1.70, "draw": 3.80, "away": 5.40},
    )
    csv_text = """snapshot_id,match_id,home_team,away_team,snapshot_at,available_at,odds_type,bookmaker,home_odds,draw_odds,away_odds,book_count,source_url
open-72,fixture-001,Norway,Iraq,2026-06-17T20:00:00,2026-06-17T20:05:00,open,book-a,1.65,3.90,5.80,1,https://example.test/open
snap-48,fixture-001,Norway,Iraq,2026-06-18T19:30:00,2026-06-18T19:35:00,current,book-a,1.58,4.00,6.20,1,https://example.test/t48
late-48,fixture-001,Norway,Iraq,2026-06-18T20:30:00,2026-06-18T20:35:00,current,book-a,1.40,4.50,8.20,1,https://example.test/late48
snap-24,fixture-001,Norway,Iraq,2026-06-19T19:30:00,2026-06-19T19:35:00,current,book-a,1.52,4.10,6.80,1,https://example.test/t24
snap-2,fixture-001,Norway,Iraq,2026-06-20T17:30:00,2026-06-20T17:35:00,current,book-a,1.48,4.20,7.20,1,https://example.test/t2
snap-1,fixture-001,Norway,Iraq,2026-06-20T18:50:00,2026-06-20T18:55:00,current,book-a,1.46,4.30,7.40,1,https://example.test/t1
closing,fixture-001,Norway,Iraq,2026-06-20T20:00:00,2026-06-20T20:05:00,closing,book-a,1.35,4.80,9.00,1,https://example.test/closing
"""
    snapshots = load_odds_snapshot_csv_text(csv_text)
    assert len(snapshots) == 7
    csv_audit = audit_odds_snapshot_csv_text(csv_text)
    assert csv_audit.quality_gate == "fail"
    assert "license_note" in csv_audit.missing_required_columns

    compliant_csv_text = """snapshot_id,match_id,home_team,away_team,kickoff_at,snapshot_at,available_at,odds_type,bookmaker,home_odds,draw_odds,away_odds,book_count,source_url,license_note
snap-48,fixture-001,Norway,Iraq,2026-06-20T20:00:00,2026-06-18T19:30:00,2026-06-18T19:35:00,current,book-a,1.58,4.00,6.20,1,https://example.test/t48,Synthetic local fixture
snap-24,fixture-001,Norway,Iraq,2026-06-20T20:00:00,2026-06-19T19:30:00,2026-06-19T19:35:00,current,book-a,1.52,4.10,6.80,1,https://example.test/t24,Synthetic local fixture
snap-2,fixture-001,Norway,Iraq,2026-06-20T20:00:00,2026-06-20T17:30:00,2026-06-20T17:35:00,current,book-a,1.48,4.20,7.20,1,https://example.test/t2,Synthetic local fixture
snap-1,fixture-001,Norway,Iraq,2026-06-20T20:00:00,2026-06-20T18:50:00,2026-06-20T18:55:00,current,book-a,1.46,4.30,7.40,1,https://example.test/t1,Synthetic local fixture
"""
    compliant_audit = audit_odds_snapshot_csv_text(compliant_csv_text)
    assert compliant_audit.quality_gate == "pass"
    assert compliant_audit.release_gate == "allows_1_0"

    assert prediction_deadline_for_horizon(kickoff, "T-48h") == datetime.fromisoformat(
        "2026-06-18T20:00:00"
    )
    t48 = select_odds_snapshot_for_horizon(record, snapshots, "T-48h")
    assert t48 is not None
    assert t48.snapshot_id == "snap-48"
    assert t48.home_odds == 1.58

    t24 = select_odds_snapshot_for_horizon(record, snapshots, "T-24h")
    assert t24 is not None
    assert t24.snapshot_id == "snap-24"

    t2 = select_odds_snapshot_for_horizon(record, snapshots, "T-2h")
    assert t2 is not None
    assert t2.snapshot_id == "snap-2"

    t1 = select_odds_snapshot_for_horizon(record, snapshots, "T-1h")
    assert t1 is not None
    assert t1.snapshot_id == "snap-1"
    assert select_odds_snapshot_for_horizon(rematch, snapshots, "T-1h") is None

    selection = materialize_horizon_records([record], snapshots)
    assert selection.selected_count == 4
    assert selection.missing_count == 0
    assert selection.leakage_issues == 0
    assert selection.horizon_counts == {"T-48h": 1, "T-24h": 1, "T-2h": 1, "T-1h": 1}
    assert {item.odds_snapshot_type for item in selection.records} == {"T-48h", "T-24h", "T-2h", "T-1h"}
    assert all(not item.leakage_issues() for item in selection.records)

    base = FootballBacktestDataset(
        manifest=DatasetManifest(
            dataset_id="fixture_base",
            source_url="https://example.test/fixture",
            license_note="Synthetic fixture for smoke testing.",
            competition="fixture_cup",
            season="2026",
            case_count=1,
        ),
        records=[record],
    )
    dataset = build_horizon_odds_dataset(base, snapshots)
    assert dataset.manifest.horizon_policy == "T-48h,T-24h,T-2h,T-1h"
    assert dataset.manifest.coverage["horizon_records"] == 4
    assert dataset.manifest.coverage["horizon_leakage_issues"] == 0
    coverage_audit = audit_horizon_coverage(base, snapshots)
    assert coverage_audit.quality_gate == "pass"
    assert coverage_audit.release_gate == "allows_1_0"
    assert coverage_audit.coverage_rate == 1.0
    assert coverage_audit.failed_slices == []
    assert coverage_audit.slice_coverage["fixture_cup:2026:T-48h"]["coverage_rate"] == 1.0
    incomplete_base = FootballBacktestDataset(
        manifest=DatasetManifest(
            dataset_id="fixture_base_incomplete",
            source_url="https://example.test/fixture",
            license_note="Synthetic fixture for smoke testing.",
            competition="fixture_cup",
            season="2026",
            case_count=2,
        ),
        records=[record, rematch],
    )
    incomplete_audit = audit_horizon_coverage(incomplete_base, snapshots)
    assert incomplete_audit.quality_gate == "fail"
    assert incomplete_audit.release_gate == "blocks_1_0"
    assert len(incomplete_audit.failed_slices) == 4
    assert incomplete_audit.slice_coverage["fixture_cup:2026:T-1h"]["coverage_rate"] == 0.5
    print(
        {
            "snapshots": len(snapshots),
            "selected": selection.selected_count,
            "horizon_counts": selection.horizon_counts,
            "dataset_manifest": asdict(dataset.manifest),
        }
    )


if __name__ == "__main__":
    main()
