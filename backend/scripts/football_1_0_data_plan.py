from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.data_sources.football_dataset_registry import (
    football_1_0_dataset_matrix,
    required_feature_coverage,
)


def write_markdown(matrix: list[dict], coverage: dict[str, list[str]], output_dir: Path) -> Path:
    lines = [
        "# Football 1.0 Data Coverage Plan",
        "",
        "This plan lists public/free or optional manually supplied datasets that can move the project from a research alpha toward football-1.0. It is a planning artifact, not a claim that every source is already integrated into the runtime.",
        "",
        "## Dataset Matrix",
        "",
        "| Priority | Dataset | Competitions | Feature Targets | Backtest Use | Main Risks |",
        "|---:|---|---|---|---|---|",
    ]
    for item in matrix:
        lines.append(
            f"| {item['priority']} | {item['label']} | {', '.join(item['competitions'])} | "
            f"{', '.join(item['feature_targets'])} | {', '.join(item['backtest_use'])} | "
            f"{'; '.join(item['risks']) or 'None listed'} |"
        )
    lines.extend(
        [
            "",
            "## Feature Coverage",
            "",
            "| Feature Area | Candidate Datasets |",
            "|---|---|",
        ]
    )
    for feature, datasets in sorted(coverage.items()):
        lines.append(f"| {feature} | {', '.join(datasets)} |")
    lines.extend(
        [
            "",
            "## 1.0 Execution Order",
            "",
            "1. Expand club-league backtesting with Football-Data.co.uk 1X2 odds and results.",
            "2. Expand national-team backtesting with openfootball World Cup and Euro fixtures plus World Football Elo snapshots.",
            "3. Use StatsBomb Open Data only as a training/evaluation source for xG, lineup continuity, and tactical style; never leak post-match events into prematch predictions.",
            "4. Add Open-Meteo forecast/historical handling with horizon-safe timestamps.",
            "5. Keep optional/manual datasets behind explicit user download/license review gates.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "football_1_0_data_coverage_plan.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    output_dir = ROOT / "outputs"
    matrix = football_1_0_dataset_matrix()
    coverage = required_feature_coverage()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "football_1_0_data_coverage_plan.json"
    json_path.write_text(
        json.dumps({"datasets": matrix, "feature_coverage": coverage}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path = write_markdown(matrix, coverage, output_dir)
    print(
        json.dumps(
            {
                "datasets": len(matrix),
                "feature_areas": len(coverage),
                "json": str(json_path),
                "markdown": str(md_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
