from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.data_sources.odds_snapshot_importers.csv_normalizer import (  # noqa: E402
    normalize_odds_snapshot_csv_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a legally obtained/public football odds CSV into the project's "
            "canonical odds_snapshot_csv contract."
        )
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--source-url-default", required=True)
    parser.add_argument("--license-note", required=True)
    parser.add_argument("--bookmaker-default", default="unknown")
    parser.add_argument("--odds-type-default", default="current")
    parser.add_argument("--mapping-json", default=None)
    parser.add_argument("--audit-json", default=None)
    args = parser.parse_args()

    mapping = None
    if args.mapping_json:
        mapping_path = Path(args.mapping_json)
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        if not isinstance(mapping, dict):
            raise SystemExit("--mapping-json must contain an object mapping canonical columns to input columns")

    result = normalize_odds_snapshot_csv_path(
        Path(args.input_csv),
        output_path=Path(args.output_csv),
        source_url_default=args.source_url_default,
        license_note=args.license_note,
        mapping={str(key): str(value) for key, value in mapping.items()} if mapping else None,
        bookmaker_default=args.bookmaker_default,
        odds_type_default=args.odds_type_default,
    )
    payload = {
        "input_csv": args.input_csv,
        "output_csv": args.output_csv,
        "row_count": result.row_count,
        "audit": asdict(result.audit),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.audit_json:
        audit_path = Path(args.audit_json)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
