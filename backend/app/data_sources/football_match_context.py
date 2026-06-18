from __future__ import annotations

from typing import Any

from app.core.entities import infer_competitors
from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest, StructuredFootballFeature


class FootballMatchContextSource:
    """Structured user-provided match context for lineup and referee data.

    Public sources often publish confirmed lineups and referee appointments
    close to kickoff. Until a dedicated official API is connected, this source
    accepts explicit context fields and records gaps when they are missing.
    """

    name = "football_match_context"

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if request.domain != "football":
            return []

        items: list[EvidenceItem] = []
        structured_features = self._structured_features(request)
        lineup_claim = self._lineup_claim(request.context)
        availability_claim = self._availability_claim(request.context)
        if lineup_claim or availability_claim or any(f.impact_area == "lineup_availability" for f in structured_features):
            claim = " ".join(part for part in [lineup_claim, availability_claim] if part)
            if not claim:
                claim = "Structured lineup and availability context supplied."
            items.append(
                self._item(
                    claim,
                    "lineup_availability",
                    0.78,
                    0.86,
                    [f for f in structured_features if f.impact_area == "lineup_availability"],
                )
            )
        else:
            items.append(self._gap("Confirmed or predicted lineup context missing.", "lineup_availability"))

        referee = request.context.get("referee")
        if referee:
            items.append(
                self._item(
                    f"Referee context provided: {referee}.",
                    "referee_environment",
                    0.72,
                    0.78,
                    [f for f in structured_features if f.impact_area == "referee_environment"],
                )
            )
        else:
            items.append(self._gap("Referee appointment context missing or not public yet.", "referee_environment"))
        return items

    def _lineup_claim(self, context: dict[str, Any]) -> str | None:
        lineups = context.get("confirmed_lineups") or context.get("predicted_lineups") or context.get("lineups")
        injuries = context.get("injuries") or context.get("suspensions")
        parts: list[str] = []
        if lineups:
            parts.append(f"Lineup context provided: {self._compact(lineups)}")
        if injuries:
            parts.append(f"Availability context provided: {self._compact(injuries)}")
        return " ".join(parts) if parts else None

    def _availability_claim(self, context: dict[str, Any]) -> str | None:
        availability = context.get("availability_context")
        if availability:
            return f"Structured availability context provided: {self._compact(availability)}"
        lineup_context = context.get("lineup_context")
        if lineup_context:
            return f"Structured lineup context provided: {self._compact(lineup_context)}"
        return None

    def _structured_features(self, request: PredictionRequest) -> list[StructuredFootballFeature]:
        first, second = infer_competitors(request)
        if not first or not second:
            return []
        features = []
        features.extend(self._availability_features(request, first, second))
        lineup_feature = self._lineup_feature(request, first, second)
        if lineup_feature:
            features.append(lineup_feature)
        return features

    def _availability_features(
        self,
        request: PredictionRequest,
        first: str,
        second: str,
    ) -> list[StructuredFootballFeature]:
        raw = request.context.get("availability_context") or []
        if isinstance(raw, dict):
            raw = raw.get("items") or raw.get("players") or []
        if not isinstance(raw, list):
            return []

        features = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            team = str(item.get("team") or "")
            normalized_team = self._normalize(team)
            if normalized_team not in {self._normalize(first), self._normalize(second)}:
                continue
            status = str(item.get("status") or "unknown").lower()
            if status in {"available", "fit", "starting", "bench"}:
                continue
            try:
                expected_minutes = float(item.get("expected_minutes_share", 0.5))
                importance = float(item.get("player_importance", 0.5))
                scarcity = float(item.get("position_scarcity", 0.5))
                replacement = float(item.get("replacement_quality", 0.5))
                certainty = float(item.get("certainty", 0.6))
            except (TypeError, ValueError):
                continue
            impact = expected_minutes * importance * scarcity * max(1 - replacement, 0) * certainty
            impact = max(min(impact, 1), 0)
            favors_first = normalized_team == self._normalize(second)
            feature_type = "suspension" if "suspend" in status else "injury"
            features.append(
                StructuredFootballFeature(
                    feature_type=feature_type,
                    impact_area="lineup_availability",
                    feature_value={
                        "status": status,
                        "expected_minutes_share": expected_minutes,
                        "player_importance": importance,
                        "position_scarcity": scarcity,
                        "replacement_quality": replacement,
                        "availability_impact": impact,
                    },
                    direction=1.0 if favors_first else -1.0,
                    magnitude=impact,
                    confidence=max(min(certainty, 1), 0),
                    feature_confidence=max(min(certainty * 0.86, 1), 0),
                    team=team,
                    player=str(item.get("player") or item.get("player_name") or ""),
                    available_at=request.prediction_deadline,
                    prediction_deadline=request.prediction_deadline,
                    leakage_risk="medium",
                    extraction_method="availability_context",
                    source_name=self.name,
                    rationale=(
                        f"Structured {feature_type} availability impact for {team} "
                        f"has modeled impact {impact:.2f}."
                    ),
                )
            )
        return features

    def _lineup_feature(
        self,
        request: PredictionRequest,
        first: str,
        second: str,
    ) -> StructuredFootballFeature | None:
        raw = request.context.get("lineup_context")
        if isinstance(raw, dict):
            rows = raw.get("teams") or raw.get("items") or [raw]
        else:
            rows = raw if isinstance(raw, list) else []
        if not isinstance(rows, list):
            return None

        by_team = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            team = self._normalize(str(row.get("team") or ""))
            if team:
                by_team[team] = row
        first_row = by_team.get(self._normalize(first))
        second_row = by_team.get(self._normalize(second))
        if not first_row or not second_row:
            return None

        def score(row: dict[str, Any]) -> float:
            try:
                xi_strength = float(row.get("xi_strength", 0.5))
                continuity = float(row.get("continuity_score", 0.5))
                surprise = float(row.get("negative_surprise_score", row.get("surprise_score", 0.0)))
            except (TypeError, ValueError):
                return 0.0
            return xi_strength * 0.65 + continuity * 0.2 - surprise * 0.15

        first_score = score(first_row)
        second_score = score(second_row)
        delta = max(min(first_score - second_score, 1), -1)
        lineup_type = str(first_row.get("lineup_type") or second_row.get("lineup_type") or "predicted_lineup")
        feature_type = "confirmed_lineup" if "confirm" in lineup_type else "predicted_lineup"
        certainty = min(float(first_row.get("certainty", 0.65)), float(second_row.get("certainty", 0.65)))
        return StructuredFootballFeature(
            feature_type=feature_type,
            impact_area="lineup_availability",
            feature_value={
                "first_score": first_score,
                "second_score": second_score,
                "lineup_delta": delta,
            },
            direction=1.0 if delta >= 0 else -1.0,
            magnitude=abs(delta),
            confidence=max(min(certainty, 1), 0),
            feature_confidence=max(min(certainty * 0.82, 1), 0),
            available_at=request.prediction_deadline,
            prediction_deadline=request.prediction_deadline,
            leakage_risk="low" if feature_type == "predicted_lineup" else "medium",
            extraction_method="lineup_context",
            source_name=self.name,
            rationale=f"Structured {feature_type} strength and continuity delta is {delta:.2f}.",
        )

    def _compact(self, value: Any) -> str:
        text = str(value)
        return " ".join(text.split())[:700]

    def _item(
        self,
        claim: str,
        impact_area: str,
        reliability: float,
        confidence: float,
        structured_features: list[StructuredFootballFeature] | None = None,
    ) -> EvidenceItem:
        return EvidenceItem(
            claim=claim,
            source=self.name,
            source_query="request.context",
            evidence_stage="verified_candidate",
            raw_excerpt=claim,
            verifier_notes=["Structured context supplied with prediction request."],
            impact_area=impact_area,
            source_reliability=reliability,
            recency_score=0.9,
            corroboration_count=0,
            confidence=confidence,
            structured_features=structured_features or [],
        )

    def _gap(self, message: str, impact_area: str) -> EvidenceItem:
        return EvidenceItem(
            claim=message,
            source=self.name,
            source_query="request.context",
            evidence_stage="collection_gap",
            impact_area=impact_area,
            source_reliability=0.7,
            recency_score=0.0,
            confidence=0.0,
        )

    def _normalize(self, value: str) -> str:
        return " ".join(value.lower().split())
