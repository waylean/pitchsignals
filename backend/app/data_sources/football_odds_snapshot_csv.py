from __future__ import annotations

import csv
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import httpx

from app.core.entities import infer_competitors
from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest, StructuredFootballFeature


class FootballOddsSnapshotCsvSource:
    """Structured public/free odds snapshot adapter.

    Accepts one of:
    - context["odds_snapshot_csv_url"]
    - context["odds_snapshot_csv_path"]
    - context["odds_snapshot_csv_text"]
    - context["odds_snapshots"] as a list of dict rows
    """

    name = "football_odds_snapshot_csv"

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if request.domain != "football":
            return []
        rows, source_url, error = await self._rows(request.context)
        if error:
            return [self._gap(error)]
        if not rows:
            return [self._gap("No odds snapshot CSV rows supplied.")]

        first, second = infer_competitors(request)
        if not first or not second:
            return [self._gap("Odds snapshot CSV requires two competitors.")]
        row = self._find_row(
            rows,
            first,
            second,
            request.context.get("match_id"),
            request.prediction_deadline,
        )
        if not row:
            return [self._gap(f"No legal odds snapshot row matched {first} vs {second}.")]

        parsed = self._parse_row(row, first, second)
        if not parsed:
            return [self._gap("Matched odds snapshot row has invalid 1X2 decimal odds.")]

        snapshot_at = self._parse_datetime(row.get("snapshot_at") or row.get("as_of"))
        available_at = self._parse_datetime(row.get("available_at") or row.get("as_of") or row.get("snapshot_at"))
        if request.prediction_deadline:
            if available_at and self._after(available_at, request.prediction_deadline):
                return [self._gap("Odds snapshot was available after the prediction deadline.")]
            if snapshot_at and self._after(snapshot_at, request.prediction_deadline):
                return [self._gap("Odds snapshot was recorded after the prediction deadline.")]

        feature = StructuredFootballFeature(
            feature_type="odds",
            impact_area="market_odds",
            feature_value=parsed,
            direction=parsed["direction"],
            magnitude=parsed["magnitude"],
            confidence=0.88,
            feature_confidence=0.88,
            match_id=str(row.get("match_id") or request.context.get("match_id") or ""),
            snapshot_at=snapshot_at,
            available_at=available_at,
            prediction_deadline=request.prediction_deadline,
            leakage_risk="low" if snapshot_at and available_at else "medium",
            extraction_method="odds_snapshot_csv",
            source_name=self.name,
            source_url=source_url or row.get("source_url"),
            license_note=str(row.get("license_note") or ""),
            source_provenance={
                "source_url": source_url or row.get("source_url"),
                "license_note": row.get("license_note"),
                "snapshot_id": row.get("snapshot_id"),
                "match_id": row.get("match_id") or request.context.get("match_id"),
            },
            rationale=(
                "Structured 1X2 odds snapshot gives "
                f"{first} {parsed['first_prob']:.1%}, draw {parsed['draw_prob']:.1%}, "
                f"{second} {parsed['second_prob']:.1%} after overround adjustment."
            ),
        )
        claim = (
            f"Structured odds snapshot: {first} {parsed['first_odds']:.2f}, draw {parsed['draw_odds']:.2f}, "
            f"{second} {parsed['second_odds']:.2f}; probabilities home={parsed['first_prob']:.3f}, "
            f"draw={parsed['draw_prob']:.3f}, away={parsed['second_prob']:.3f}; "
            f"snapshot_at={row.get('snapshot_at') or row.get('as_of') or 'unknown'}; "
            f"available_at={row.get('available_at') or row.get('as_of') or row.get('snapshot_at') or 'unknown'}."
        )
        return [
            EvidenceItem(
                claim=claim,
                source=self.name,
                source_url=source_url or row.get("source_url"),
                license_note=str(row.get("license_note") or ""),
                source_provenance={
                    "source_url": source_url or row.get("source_url"),
                    "license_note": row.get("license_note"),
                    "snapshot_id": row.get("snapshot_id"),
                    "match_id": row.get("match_id") or request.context.get("match_id"),
                },
                source_query="odds snapshot CSV",
                evidence_stage="verified_candidate",
                raw_excerpt=claim,
                verifier_notes=["Parsed from structured odds snapshot CSV/context."],
                published_at=available_at,
                impact_area="market_odds",
                source_reliability=0.88,
                recency_score=0.82,
                corroboration_count=int(parsed.get("book_count") or 1),
                confidence=0.88,
                structured_features=[feature],
            )
        ]

    async def _rows(self, context: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None, str | None]:
        supplied = context.get("odds_snapshots")
        if isinstance(supplied, list):
            return [row for row in supplied if isinstance(row, dict)], None, None

        text = context.get("odds_snapshot_csv_text")
        source_url = None
        if not isinstance(text, str):
            url = context.get("odds_snapshot_csv_url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                try:
                    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                        response = await client.get(url)
                        response.raise_for_status()
                    text = response.text
                    source_url = url
                except Exception as exc:
                    return [], None, f"Odds snapshot CSV download failed: {exc}"

        if not isinstance(text, str):
            path = context.get("odds_snapshot_csv_path")
            if isinstance(path, str):
                try:
                    text = Path(path).read_text(encoding="utf-8")
                    source_url = str(Path(path).resolve())
                except OSError as exc:
                    return [], None, f"Odds snapshot CSV file read failed: {exc}"

        if not isinstance(text, str) or not text.strip():
            return [], None, None
        return list(csv.DictReader(StringIO(text))), source_url, None

    def _find_row(
        self,
        rows: list[dict[str, Any]],
        first: str,
        second: str,
        match_id: Any,
        prediction_deadline: datetime | None = None,
    ) -> dict[str, Any] | None:
        normalized_first = self._normalize(first)
        normalized_second = self._normalize(second)
        candidates = []
        for row in rows:
            if match_id and str(row.get("match_id")) == str(match_id):
                matched = True
            else:
                home = self._normalize(str(row.get("home_team") or row.get("team1") or ""))
                away = self._normalize(str(row.get("away_team") or row.get("team2") or ""))
                matched = home == normalized_first and away == normalized_second
            if not matched:
                continue
            snapshot_at = self._parse_datetime(row.get("snapshot_at") or row.get("as_of"))
            available_at = self._parse_datetime(row.get("available_at") or row.get("as_of") or row.get("snapshot_at"))
            if prediction_deadline:
                if snapshot_at and self._after(snapshot_at, prediction_deadline):
                    continue
                if available_at and self._after(available_at, prediction_deadline):
                    continue
            candidates.append((available_at or snapshot_at or datetime.min, snapshot_at or datetime.min, row))
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: (item[0], item[1]))[-1][2]

    def _parse_row(self, row: dict[str, Any], first: str, second: str) -> dict[str, float] | None:
        try:
            home_odds = float(row.get("home_odds") or row.get("home") or row.get("h"))
            draw_odds = float(row.get("draw_odds") or row.get("draw") or row.get("d"))
            away_odds = float(row.get("away_odds") or row.get("away") or row.get("a"))
        except (TypeError, ValueError):
            return None
        if min(home_odds, draw_odds, away_odds) <= 1:
            return None

        raw_home = 1 / home_odds
        raw_draw = 1 / draw_odds
        raw_away = 1 / away_odds
        total = raw_home + raw_draw + raw_away
        home_prob = raw_home / total
        draw_prob = raw_draw / total
        away_prob = raw_away / total

        home = self._normalize(str(row.get("home_team") or row.get("team1") or ""))
        first_is_home = self._normalize(first) == home
        first_prob = home_prob if first_is_home else away_prob
        second_prob = away_prob if first_is_home else home_prob
        first_odds = home_odds if first_is_home else away_odds
        second_odds = away_odds if first_is_home else home_odds
        value = (first_prob - second_prob) / max(first_prob + second_prob, 0.001)
        return {
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "first_prob": first_prob,
            "second_prob": second_prob,
            "first_odds": first_odds,
            "draw_odds": draw_odds,
            "second_odds": second_odds,
            "overround": total - 1,
            "direction": 1.0 if value >= 0 else -1.0,
            "magnitude": min(abs(value), 1.0),
            "book_count": float(row.get("book_count") or 1),
            "odds_type": str(row.get("odds_type") or row.get("snapshot_type") or "snapshot"),
            "bookmaker": str(row.get("bookmaker") or row.get("source") or self.name),
            "source_url": str(row.get("source_url") or ""),
        }

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        text = str(value).strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M", "%d/%m/%Y"]:
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None

    def _after(self, left: datetime, right: datetime) -> bool:
        try:
            return left > right
        except TypeError:
            return left.replace(tzinfo=None) > right.replace(tzinfo=None)

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().replace("&", "and").split())

    def _gap(self, message: str) -> EvidenceItem:
        return EvidenceItem(
            claim=message,
            source=self.name,
            source_query="odds snapshot CSV",
            evidence_stage="collection_gap",
            impact_area="market_odds",
            source_reliability=0.82,
            recency_score=0.0,
            confidence=0.0,
        )
