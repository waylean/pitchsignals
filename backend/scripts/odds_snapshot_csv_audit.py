from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.backtesting.datasets import load_football_data_dataset  # noqa: E402
from app.backtesting.odds_snapshots import (  # noqa: E402
    audit_horizon_coverage,
    audit_odds_snapshot_csv_path,
    load_odds_snapshot_csv_path,
)
from app.backtesting.records import DatasetManifest, FootballBacktestDataset  # noqa: E402


DEFAULT_LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--odds-snapshot-csv-path", required=True)
    parser.add_argument("--leagues", nargs="*", default=None)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--horizons", nargs="*", default=["T-48h", "T-24h", "T-2h", "T-1h"])
    parser.add_argument("--min-coverage-rate", type=float, default=0.95)
    parser.add_argument("--min-slice-coverage-rate", type=float, default=0.95)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    csv_path = Path(args.odds_snapshot_csv_path)
    csv_audit = audit_odds_snapshot_csv_path(csv_path)
    result = {
        "csv_audit": asdict(csv_audit),
        "horizon_coverage_audit": None,
    }
    if args.leagues and args.seasons:
        datasets = []
        failures = []
        cache_dir = ROOT / "work" / "football_data_cache"
        for season in args.seasons:
            for league in args.leagues:
                try:
                    datasets.append(await load_football_data_dataset(cache_dir, league=league, season=season))
                except Exception as exc:
                    failures.append({"league": league, "season": season, "error": str(exc)})
        records = [record for dataset in datasets for record in dataset.records]
        base = FootballBacktestDataset(
            manifest=DatasetManifest(
                dataset_id="odds_snapshot_audit_base",
                source_url="https://www.football-data.co.uk/data.php",
                license_note=(
                    "Football-Data.co.uk public CSV. Do not redistribute the full dataset in-repo; "
                    "cache locally and attribute source."
                ),
                competition=",".join(args.leagues or DEFAULT_LEAGUES),
                season=",".join(args.seasons),
                case_count=len(records),
                snapshot_policy="downloaded_csv_cache",
                horizon_policy="market_snapshot_unknown_or_closing_proxy",
                license_tier="public_free_attributed",
                redistribution_allowed=False,
                coverage={
                    "datasets_loaded": len(datasets),
                    "datasets_failed": len(failures),
                    "records_with_1x2_odds": sum(1 for record in records if record.has_1x2_odds),
                },
                gaps=[json.dumps(item, ensure_ascii=False) for item in failures[:20]],
            ),
            records=records,
        )
        snapshots = load_odds_snapshot_csv_path(csv_path)
        horizon_audit = audit_horizon_coverage(
            base,
            snapshots,
            horizons=args.horizons,
            min_coverage_rate=args.min_coverage_rate,
            min_slice_coverage_rate=args.min_slice_coverage_rate,
        )
        result["horizon_coverage_audit"] = asdict(horizon_audit)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
