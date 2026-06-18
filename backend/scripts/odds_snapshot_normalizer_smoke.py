from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.backtesting.odds_snapshots import load_odds_snapshot_csv_text  # noqa: E402
from app.data_sources.odds_snapshot_importers.csv_normalizer import (  # noqa: E402
    CANONICAL_ODDS_SNAPSHOT_COLUMNS,
    normalize_odds_snapshot_csv_text,
)


def main() -> None:
    raw = """Date,HomeTeam,AwayTeam,as_of,B365H,B365D,B365A
2024-08-16,Man United,Fulham,2024-08-14T12:00:00,1.62,4.10,5.80
2024-08-17,Ipswich,Liverpool,2024-08-15T13:30:00,8.00,5.10,1.38
"""
    result = normalize_odds_snapshot_csv_text(
        raw,
        source_url_default="https://example.test/legal-export.csv",
        license_note="Synthetic smoke file representing a legally obtained public/free export.",
        bookmaker_default="B365",
        odds_type_default="current",
        source="normalizer_smoke",
    )
    assert result.row_count == 2
    assert result.audit.quality_gate == "pass"
    assert result.audit.release_gate == "allows_1_0"
    header = result.text.splitlines()[0].split(",")
    assert header == CANONICAL_ODDS_SNAPSHOT_COLUMNS
    snapshots = load_odds_snapshot_csv_text(result.text, source="normalizer_smoke")
    assert len(snapshots) == 2
    assert snapshots[0].home_team == "Man United"
    assert snapshots[0].bookmaker == "B365"
    assert snapshots[0].source_url == "https://example.test/legal-export.csv"
    print(
        {
            "rows": result.row_count,
            "quality_gate": result.audit.quality_gate,
            "release_gate": result.audit.release_gate,
            "snapshots": len(snapshots),
        }
    )


if __name__ == "__main__":
    main()
