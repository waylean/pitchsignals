from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.backtesting.datasets import load_football_data_dataset
from app.backtesting.reports import write_club_backtest_report
from app.backtesting.runner import football_data_market_backtest


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="premier_league")
    parser.add_argument("--season", default="2324")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    dataset = await load_football_data_dataset(
        ROOT / "work" / "football_data_cache",
        league=args.league,
        season=args.season,
    )
    result = football_data_market_backtest(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{dataset.manifest.dataset_id}_backtest.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_club_backtest_report(result, output_dir)
    print(
        json.dumps(
            {
                "dataset_id": dataset.manifest.dataset_id,
                "cases": dataset.manifest.case_count,
                "leaderboard_top": result["leaderboard"]["entries"][:3],
                "promotion_decision": result["promotion_decision"],
                "json": str(json_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
