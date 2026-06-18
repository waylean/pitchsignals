from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_club_backtest_report(result: dict[str, Any], output_dir: Path) -> Path:
    manifest = result["manifest"]
    leaderboard = result["leaderboard"]["entries"]
    promotion = result["promotion_decision"]
    lines = [
        "# Football-Data Club League Backtest",
        "",
        "## Scope",
        "",
        f"- Dataset: `{manifest['dataset_id']}`",
        f"- Source: `{manifest['source_url']}`",
        f"- Competition: `{manifest.get('competition')}`",
        f"- Season: `{manifest.get('season')}`",
        f"- Cases: {manifest['case_count']}",
        f"- License note: {manifest['license_note']}",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model/Profile | Log loss | Brier | RPS | Accuracy | Cases |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for entry in leaderboard:
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
            json.dumps(promotion, indent=2),
            "```",
            "",
            "## Sample Matches",
            "",
            "| Date | Match | Score | Actual | Odds H/D/A |",
            "|---|---|---|---|---|",
        ]
    )
    for row in result["sample_matches"]:
        odds = row["odds"]
        lines.append(
            f"| {row['date']} | {row['match']} | {row['score']} | {row['actual']} | "
            f"{odds['home']:.2f}/{odds['draw']:.2f}/{odds['away']:.2f} |"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{manifest['dataset_id']}_backtest_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
