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

from app.backtesting.datasets import load_football_data_dataset
from app.backtesting.records import DatasetManifest, FootballBacktestDataset
from app.backtesting.runner import football_data_sliced_market_backtest


DEFAULT_LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
DEFAULT_SEASONS = ["2122", "2223", "2324"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES)
    parser.add_argument("--seasons", nargs="*", default=DEFAULT_SEASONS)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()

    cache_dir = ROOT / "work" / "football_data_cache"
    datasets = []
    failures = []
    for season in args.seasons:
        for league in args.leagues:
            try:
                datasets.append(await load_football_data_dataset(cache_dir, league=league, season=season))
            except Exception as exc:
                failures.append({"league": league, "season": season, "error": str(exc)})

    records = [record for dataset in datasets for record in dataset.records]
    manifest = DatasetManifest(
        dataset_id="football_data_multi_league_multi_season",
        source_url="https://www.football-data.co.uk/data.php",
        license_note=(
            "Football-Data.co.uk public CSV. Do not redistribute the full dataset in-repo; "
            "cache locally and attribute source."
        ),
        competition=",".join(args.leagues),
        season=",".join(args.seasons),
        case_count=len(records),
        coverage={
            "datasets_loaded": len(datasets),
            "datasets_failed": len(failures),
            "leagues": ",".join(args.leagues),
            "seasons": ",".join(args.seasons),
            "records_with_1x2_odds": sum(1 for record in records if record.has_1x2_odds),
        },
        gaps=[json.dumps(item, ensure_ascii=False) for item in failures[:20]],
    )
    combined = FootballBacktestDataset(manifest=manifest, records=records)
    result = football_data_sliced_market_backtest(combined)
    result["dataset_manifests"] = [asdict(dataset.manifest) for dataset in datasets]
    result["failures"] = failures

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "football_data_multi_league_backtest.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_multi_report(result, output_dir)
    print(
        json.dumps(
            {
                "datasets_loaded": len(datasets),
                "failures": failures,
                "cases": len(records),
                "leaderboard_top": result["leaderboard"]["entries"][:3],
                "promotion_decision": result["promotion_decision"],
                "json": str(json_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def write_multi_report(result: dict, output_dir: Path) -> Path:
    manifest = result["manifest"]
    lines = [
        "# Football-Data Multi-League Backtest",
        "",
        "## Scope",
        "",
        f"- Dataset: `{manifest['dataset_id']}`",
        f"- Competitions: `{manifest.get('competition')}`",
        f"- Seasons: `{manifest.get('season')}`",
        f"- Cases: {manifest['case_count']}",
        f"- Datasets loaded: {manifest['coverage']['datasets_loaded']}",
        f"- Datasets failed: {manifest['coverage']['datasets_failed']}",
        "",
        "## Overall Leaderboard",
        "",
        "| Rank | Model/Profile | Log loss | Brier | RPS | Accuracy | Cases |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for entry in result["leaderboard"]["entries"]:
        lines.append(
            f"| {entry['rank']} | {entry['model_id']} | {entry['log_loss']:.4f} | "
            f"{entry['brier_score']:.4f} | {entry['ranked_probability_score']:.4f} | "
            f"{entry['accuracy']:.3f} | {entry['case_count']} |"
        )
    lines.extend(
        [
            "",
            "## Promotion Decision",
            "",
            "```json",
            json.dumps(result["promotion_decision"], indent=2),
            "```",
            "",
            "## Slice Summary",
            "",
            "| Slice | Cases | Market log loss | Uniform log loss | Decision |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for slice_result in result["slices"]:
        slice_manifest = slice_result["manifest"]
        reports = {item["model_id"]: item for item in slice_result["reports"]}
        market = reports["football_data_market_only"]
        uniform = reports["uniform_baseline"]
        lines.append(
            f"| {slice_manifest['competition']} | {slice_manifest['case_count']} | "
            f"{market['log_loss']:.4f} | {uniform['log_loss']:.4f} | "
            f"{slice_result['promotion_decision']['decision']} |"
        )
    path = output_dir / "football_data_multi_league_backtest_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    asyncio.run(main())
