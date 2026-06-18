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
from app.backtesting.odds_snapshots import build_horizon_odds_dataset, load_odds_snapshot_csv_path
from app.backtesting.odds_snapshots import audit_horizon_coverage, audit_odds_snapshot_csv_path
from app.backtesting.records import DatasetManifest, FootballBacktestDataset
from app.backtesting.runner import (
    CANDIDATE_MODEL_ID,
    RAW_MARKET_MODEL_ID,
    football_data_multi_window_rolling_backtest,
    football_data_rolling_market_backtest,
)
from app.models.governance import ensemble_profile_from_dict


DEFAULT_LEAGUES = ["premier_league", "la_liga", "bundesliga", "serie_a", "ligue_1"]
DEFAULT_MULTI_WINDOW_SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leagues", nargs="*", default=DEFAULT_LEAGUES)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--train-seasons", nargs="*", default=["2122"])
    parser.add_argument("--validation-season", default="2223")
    parser.add_argument("--test-season", default="2324")
    parser.add_argument("--multi-window", action="store_true")
    parser.add_argument("--train-window-size", type=int, default=2)
    parser.add_argument("--horizon-odds", action="store_true")
    parser.add_argument("--odds-snapshot-csv-path", default=None)
    parser.add_argument("--horizons", nargs="*", default=["T-48h", "T-24h", "T-2h", "T-1h"])
    parser.add_argument("--min-horizon-coverage-rate", type=float, default=0.95)
    parser.add_argument("--min-horizon-slice-coverage-rate", type=float, default=0.95)
    parser.add_argument("--allow-horizon-audit-failures", action="store_true")
    parser.add_argument("--previous-stable-profile-path", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    args = parser.parse_args()
    if args.horizon_odds and not args.odds_snapshot_csv_path:
        parser.error("--horizon-odds requires --odds-snapshot-csv-path")

    if args.seasons:
        seasons = args.seasons
    elif args.multi_window:
        seasons = DEFAULT_MULTI_WINDOW_SEASONS
    else:
        seasons = [*args.train_seasons, args.validation_season, args.test_season]
    seasons = list(dict.fromkeys(seasons))
    cache_dir = ROOT / "work" / "football_data_cache"
    datasets = []
    failures = []
    for season in seasons:
        for league in args.leagues:
            try:
                datasets.append(await load_football_data_dataset(cache_dir, league=league, season=season))
            except Exception as exc:
                failures.append({"league": league, "season": season, "error": str(exc)})

    records = [record for dataset in datasets for record in dataset.records]
    manifest = DatasetManifest(
        dataset_id="football_data_rolling_multi_league",
        source_url="https://www.football-data.co.uk/data.php",
        license_note=(
            "Football-Data.co.uk public CSV. Do not redistribute the full dataset in-repo; "
            "cache locally and attribute source."
        ),
        competition=",".join(args.leagues),
        season=",".join(seasons),
        case_count=len(records),
        snapshot_policy="downloaded_csv_cache",
        horizon_policy="market_snapshot_unknown_or_closing_proxy",
        license_tier="public_free_attributed",
        redistribution_allowed=False,
        coverage={
            "datasets_loaded": len(datasets),
            "datasets_failed": len(failures),
            "records_with_1x2_odds": sum(1 for record in records if record.has_1x2_odds),
            "leakage_issues": sum(len(record.leakage_issues()) for record in records),
        },
        gaps=[json.dumps(item, ensure_ascii=False) for item in failures[:20]],
    )
    combined = FootballBacktestDataset(manifest=manifest, records=records)
    previous_stable_profile = None
    if args.previous_stable_profile_path:
        previous_stable_path = Path(args.previous_stable_profile_path)
        try:
            previous_stable_profile = ensemble_profile_from_dict(
                json.loads(previous_stable_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            raise SystemExit(f"Failed to load previous stable profile: {exc}") from exc
    if args.horizon_odds:
        odds_path = Path(args.odds_snapshot_csv_path)
        odds_csv_audit = audit_odds_snapshot_csv_path(odds_path)
        odds_snapshots = load_odds_snapshot_csv_path(odds_path)
        horizon_audit = audit_horizon_coverage(
            combined,
            odds_snapshots,
            horizons=args.horizons,
            min_coverage_rate=args.min_horizon_coverage_rate,
            min_slice_coverage_rate=args.min_horizon_slice_coverage_rate,
        )
        audit_failed = (
            odds_csv_audit.quality_gate == "fail"
            or horizon_audit.quality_gate == "fail"
        )
        if audit_failed and not args.allow_horizon_audit_failures:
            raise SystemExit(
                "Horizon odds audit failed. "
                f"csv_errors={odds_csv_audit.errors}; horizon_errors={horizon_audit.errors}. "
                "Use --allow-horizon-audit-failures only for local smoke fixtures."
            )
        combined = build_horizon_odds_dataset(
            combined,
            odds_snapshots,
            horizons=args.horizons,
            dataset_id="football_data_rolling_multi_league_horizon_odds",
        )
        combined.manifest.coverage.update(
            {
                "odds_csv_quality_gate": odds_csv_audit.quality_gate,
                "odds_csv_release_gate": odds_csv_audit.release_gate,
                "horizon_coverage_quality_gate": horizon_audit.quality_gate,
                "horizon_coverage_release_gate": horizon_audit.release_gate,
                "horizon_coverage_rate": horizon_audit.coverage_rate,
                "horizon_min_slice_coverage_rate": horizon_audit.min_slice_coverage_rate,
                "horizon_failed_slice_count": len(horizon_audit.failed_slices),
            }
        )
        combined.manifest.gaps.extend(
            [
                *[f"odds_csv_error:{item}" for item in odds_csv_audit.errors[:10]],
                *[f"horizon_error:{item}" for item in horizon_audit.errors[:10]],
                *[f"odds_csv_warning:{item}" for item in odds_csv_audit.warnings[:10]],
                *[f"horizon_warning:{item}" for item in horizon_audit.warnings[:10]],
            ]
        )
    if args.multi_window:
        result = football_data_multi_window_rolling_backtest(
            combined,
            seasons=seasons,
            train_window_size=args.train_window_size,
            previous_stable_profile=previous_stable_profile,
        )
    else:
        result = football_data_rolling_market_backtest(
            combined,
            train_seasons=args.train_seasons,
            validation_season=args.validation_season,
            test_season=args.test_season,
            previous_stable_profile=previous_stable_profile,
        )
    metadata = result.setdefault("metadata", {})
    metadata["dataset_manifests"] = [asdict(dataset.manifest) for dataset in datasets]
    metadata["failures"] = failures
    result["dataset_manifests"] = metadata["dataset_manifests"]
    result["failures"] = metadata["failures"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / (
        "football_data_multi_window_rolling_backtest.json"
        if args.multi_window
        else "football_data_rolling_backtest.json"
    )
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = (
        write_multi_window_report(result, output_dir)
        if args.multi_window
        else write_rolling_report(result, output_dir)
    )
    print(
        json.dumps(
            {
                "datasets_loaded": len(datasets),
                "failures": failures,
                "cases": len(records),
                "multi_window": args.multi_window,
                "summary": result.get("summary"),
                "rolling_promotion_decision": result.get("rolling_promotion_decision"),
                "split": result.get("split"),
                "market_temperature_alpha": result.get("market_temperature_alpha"),
                "test_leaderboard_top": (
                    result.get("test_leaderboard", {}).get("entries", [])[:4]
                ),
                "promotion_decision_vs_uniform": result.get("promotion_decision_vs_uniform"),
                "promotion_decision_vs_raw_market": result.get("promotion_decision_vs_raw_market"),
                "promotion_decision_vs_previous_stable": result.get(
                    "promotion_decision_vs_previous_stable"
                ),
                "json": str(json_path),
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _report_by_model(reports: list[dict], model_id: str) -> dict:
    for report in reports:
        if report["model_id"] == model_id:
            return report
    raise KeyError(model_id)


def _report_maturity(manifest: dict) -> dict[str, str]:
    horizon_policy = str(manifest.get("horizon_policy") or "")
    snapshot_policy = str(manifest.get("snapshot_policy") or "")
    coverage = manifest.get("coverage") if isinstance(manifest.get("coverage"), dict) else {}
    has_horizon_records = int(coverage.get("horizon_records") or 0) > 0
    if has_horizon_records and "T-" in horizon_policy and "available_at" in snapshot_policy:
        return {
            "report_maturity": "horizon_safe_prerelease_0.9",
            "promotion_scope": "horizon-safe candidate evaluation only; not stable/default",
            "horizon_safe": "yes",
        }
    if horizon_policy == "market_snapshot_unknown_or_closing_proxy":
        return {
            "report_maturity": "market_proxy_0.8",
            "promotion_scope": "market-proxy candidate evaluation only; not horizon-safe and not 1.0",
            "horizon_safe": "no",
        }
    return {
        "report_maturity": "research",
        "promotion_scope": "research evidence only",
        "horizon_safe": "unknown",
    }


def _append_maturity_section(lines: list[str], manifest: dict) -> None:
    maturity = _report_maturity(manifest)
    coverage = manifest.get("coverage") if isinstance(manifest.get("coverage"), dict) else {}
    lines.extend(
        [
            "## Report Maturity",
            "",
            f"- Report maturity: `{maturity['report_maturity']}`",
            f"- Promotion scope: {maturity['promotion_scope']}",
            f"- Horizon-safe replay: `{maturity['horizon_safe']}`",
            f"- Odds timestamp coverage: `{coverage.get('horizon_records', 0)}` horizon records, "
            f"`{coverage.get('horizon_leakage_issues', 0)}` horizon leakage issues",
            "",
        ]
    )


def write_rolling_report(result: dict, output_dir: Path) -> Path:
    split = result["split"]
    lines = [
        "# Football-Data Rolling Split Backtest",
        "",
        "## Scope",
        "",
        f"- Train seasons: `{', '.join(split['train_seasons'])}`",
        f"- Validation season: `{split['validation_season']}`",
        f"- Test season: `{split['test_season']}`",
        f"- Train cases: {split['train_cases']}",
        f"- Validation cases: {split['validation_cases']}",
        f"- Test cases: {split['test_cases']}",
        f"- Market calibration alpha: {result['market_temperature_alpha']:.2f}",
        f"- Horizon policy: `{result['manifest'].get('horizon_policy')}`",
        f"- Snapshot policy: `{result['manifest'].get('snapshot_policy')}`",
        "",
    ]
    _append_maturity_section(lines, result["manifest"])
    lines.extend(
        [
            "## Test Leaderboard",
            "",
            "| Rank | Model/Profile | Log loss | Brier | RPS | Accuracy | Cases |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for entry in result["test_leaderboard"]["entries"]:
        lines.append(
            f"| {entry['rank']} | {entry['model_id']} | {entry['log_loss']:.4f} | "
            f"{entry['brier_score']:.4f} | {entry['ranked_probability_score']:.4f} | "
            f"{entry['accuracy']:.3f} | {entry['case_count']} |"
        )
    lines.extend(
        [
            "",
            "## Promotion Decisions",
            "",
            "Against uniform baseline:",
            "",
            "```json",
            json.dumps(result["promotion_decision_vs_uniform"], indent=2),
            "```",
            "",
            "Against raw market-only:",
            "",
            "```json",
            json.dumps(result["promotion_decision_vs_raw_market"], indent=2),
            "```",
            "",
        ]
    )
    if result.get("promotion_decision_vs_previous_stable"):
        lines.extend(
            [
                "Against previous stable profile:",
                "",
                "```json",
                json.dumps(result["promotion_decision_vs_previous_stable"], indent=2),
                "```",
                "",
            ]
        )
    lines.extend(["## Split Metrics", ""])
    for label, key in [
        ("Train", "train_reports"),
        ("Validation", "validation_reports"),
        ("Test", "test_reports"),
    ]:
        lines.extend(
            [
                f"### {label}",
                "",
                "| Model | Log loss | Brier | RPS | Accuracy |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for report in result[key]:
            lines.append(
                f"| {report['model_id']} | {report['log_loss']:.4f} | "
                f"{report['brier_score']:.4f} | {report['ranked_probability_score']:.4f} | "
                f"{report['accuracy']:.3f} |"
            )
        lines.append("")
    if result.get("test_slice_reports"):
        lines.extend(
            [
                "## Test League Slices",
                "",
                "| Slice | Model | Log loss | Brier | RPS | Accuracy | Cases |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        candidate_model_id = result.get("candidate_model_id") or CANDIDATE_MODEL_ID
        for slice_id, reports in sorted(result["test_slice_reports"].items()):
            for model_id in [candidate_model_id, RAW_MARKET_MODEL_ID]:
                report = reports[model_id]
                lines.append(
                    f"| {slice_id} | {model_id} | {report['log_loss']:.4f} | "
                    f"{report['brier_score']:.4f} | {report['ranked_probability_score']:.4f} | "
                    f"{report['accuracy']:.3f} | {report['case_count']} |"
                )
        lines.append("")
    path = output_dir / "football_data_rolling_backtest_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_multi_window_report(result: dict, output_dir: Path) -> Path:
    summary = result["summary"]
    candidate_model_id = result.get("candidate_model_id") or CANDIDATE_MODEL_ID
    lines = [
        "# Football-Data Multi-Window Rolling Backtest",
        "",
        "## Scope",
        "",
        f"- Window count: {summary['window_count']}",
        f"- Total test cases: {summary['test_cases']}",
        f"- Horizon policy: `{result['manifest'].get('horizon_policy')}`",
        f"- Snapshot policy: `{result['manifest'].get('snapshot_policy')}`",
        "",
    ]
    _append_maturity_section(lines, result["manifest"])
    lines.extend(
        [
            "## Aggregate Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    lines.extend(
        [
        f"| Uniform-baseline promotions | {summary['uniform_promotions']} |",
        f"| Raw-market promotions | {summary['raw_market_promotions']} |",
        f"| Raw-market holds | {summary['raw_market_holds']} |",
        f"| Previous-stable promotions | {summary.get('previous_stable_promotions', 0)} |",
        f"| Previous-stable holds | {summary.get('previous_stable_holds', 0)} |",
        f"| Weighted candidate log loss | {summary['weighted_calibrated_log_loss']:.4f} |",
        f"| Weighted raw-market log loss | {summary['weighted_raw_market_log_loss']:.4f} |",
        f"| Weighted candidate Brier | {summary['weighted_calibrated_brier']:.4f} |",
        f"| Weighted raw-market Brier | {summary['weighted_raw_market_brier']:.4f} |",
        f"| Candidate model | `{candidate_model_id}` |",
        "",
        "## Rolling Promotion Decision",
        "",
        "```json",
        json.dumps(result["rolling_promotion_decision"], indent=2),
        "```",
        "",
        "## Windows",
        "",
        "| Window | Train | Validation | Test | Global alpha | League alphas | Test cases | Candidate log loss | Raw log loss | Raw-market decision | Previous-stable decision |",
        "|---|---|---|---|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for window in result["windows"]:
        split = window["split"]
        calibrated = _report_by_model(window["test_reports"], candidate_model_id)
        raw = _report_by_model(window["test_reports"], RAW_MARKET_MODEL_ID)
        league_alphas = ", ".join(
            f"{league}:{alpha:.2f}"
            for league, alpha in sorted(window.get("market_temperature_by_league", {}).items())
        )
        lines.append(
            f"| {split['split_id']} | {', '.join(split['train_seasons'])} | "
            f"{split['validation_season']} | {split['test_season']} | "
            f"{window['market_temperature_alpha']:.2f} | {league_alphas} | {split['test_cases']} | "
            f"{calibrated['log_loss']:.4f} | {raw['log_loss']:.4f} | "
            f"{window['promotion_decision_vs_raw_market']['decision']} | "
            f"{window.get('promotion_decision_vs_previous_stable', {}).get('decision', 'not_run')} |"
        )
    lines.extend(
        [
            "",
            "## Slice Holds",
            "",
            "| Window test season | Slice | Reasons |",
            "|---|---|---|",
        ]
    )
    hold_rows = 0
    for window in result["windows"]:
        test_season = window["split"]["test_season"]
        slice_results = window["promotion_decision_vs_raw_market"]["metrics"].get("slice_results", {})
        for slice_id, slice_result in sorted(slice_results.items()):
            if slice_result["decision"] != "promote":
                hold_rows += 1
                reasons = "; ".join(slice_result["reasons"])
                lines.append(f"| {test_season} | {slice_id} | {reasons} |")
    if hold_rows == 0:
        lines.append("| all | none | All required league slices passed. |")
    path = output_dir / "football_data_multi_window_rolling_backtest_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    asyncio.run(main())
