from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.core.entities import infer_competitors
from app.core.llm import get_llm_client
from app.schemas import EvidenceItem, PredictionRequest, StructuredFootballFeature


class UnofficialFootballSignalStructurer:
    """Convert noisy public snippets into structured low-confidence features.

    The output is deliberately conservative. It does not create facts; it
    transforms candidate snippets into directional hypotheses with provenance.
    """

    source_name = "ai_unofficial_signal_structurer"

    async def structure(self, request: PredictionRequest, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        if request.domain != "football" or request.context.get("skip_unofficial_ai_structuring"):
            return []
        first, second = infer_competitors(request)
        if not first or not second:
            return []

        candidates = self._candidate_items(evidence)
        if not candidates:
            return []

        client = get_llm_client()
        system = (
            "You transform noisy football snippets into structured prediction features. "
            "Do not invent facts. Treat local media, fan forums, rumors, previews, and snippets as candidate evidence only. "
            "Extract only directional hypotheses that are supported by the provided snippets. "
            "Return JSON only."
        )
        user = json.dumps(
            {
                "first_team": first,
                "second_team": second,
                "allowed_impact_areas": ["lineup_availability", "tactical_matchup", "sentiment_narrative"],
                "feature_rules": {
                    "direction": "1 favors first_team, -1 favors second_team, 0 neutral",
                    "magnitude": "0..1; keep <=0.45 unless multiple snippets agree",
                    "confidence": "0..1; unofficial evidence should usually be 0.25..0.55",
                    "feature_type": "one of rumor_availability, predicted_lineup, tactical_style, local_media_tactical, sentiment",
                },
                "snippets": candidates,
                "output_schema": {
                    "status": "ok|partial|skipped",
                    "features": [
                        {
                            "impact_area": "lineup_availability|tactical_matchup|sentiment_narrative",
                            "feature_type": "string",
                            "direction": "number -1..1",
                            "magnitude": "number 0..1",
                            "confidence": "number 0..1",
                            "rationale": "short string",
                            "supporting_indices": ["integer"],
                        }
                    ],
                },
            },
            ensure_ascii=False,
        )
        try:
            result = await client.complete_json(system, user)
        except Exception as exc:
            return [self._gap(f"AI unofficial signal structuring failed: {exc!r}")]

        features = result.get("features") if isinstance(result, dict) else None
        if not isinstance(features, list):
            return []

        now = datetime.now(timezone.utc)
        items: list[EvidenceItem] = []
        for index, feature_payload in enumerate(features[:8]):
            if not isinstance(feature_payload, dict):
                continue
            feature = self._feature_from_payload(feature_payload, request, now)
            if not feature:
                continue
            supporting = self._supporting_claims(feature_payload.get("supporting_indices"), candidates)
            claim = feature.rationale or "AI structured unofficial football signal."
            if supporting:
                claim = f"{claim} Supporting snippets: {' | '.join(supporting)[:650]}"
            items.append(
                EvidenceItem(
                    evidence_id=f"ai_unofficial_{index}",
                    claim=claim[:900],
                    source=self.source_name,
                    source_query="AI structuring of public candidate snippets",
                    evidence_stage="candidate",
                    raw_excerpt=claim[:900],
                    verifier_notes=[
                        "AI-structured unofficial candidate evidence; not treated as verified fact.",
                        "Direction and confidence are capped for rumor/local-media snippets.",
                    ],
                    collected_at=now,
                    published_at=now,
                    impact_area=feature.impact_area,
                    source_reliability=0.46,
                    recency_score=0.72,
                    corroboration_count=max(len(supporting) - 1, 0),
                    confidence=feature.confidence,
                    structured_features=[feature],
                )
            )
        return items

    def _candidate_items(self, evidence: list[EvidenceItem]) -> list[dict[str, Any]]:
        allowed = {"lineup_availability", "tactical_matchup", "sentiment_narrative"}
        rows: list[dict[str, Any]] = []
        for item in evidence:
            if item.impact_area not in allowed or item.evidence_stage not in {"candidate", "verified_candidate"}:
                continue
            if item.source in {"the_odds_api", "world_football_elo"}:
                continue
            rows.append(
                {
                    "index": len(rows),
                    "impact_area": item.impact_area,
                    "source": item.source,
                    "url": item.source_url,
                    "query": item.source_query,
                    "reliability": item.source_reliability,
                    "confidence": item.confidence,
                    "text": item.claim[:700],
                    "notes": item.verifier_notes,
                }
            )
            if len(rows) >= 18:
                break
        return rows

    def _feature_from_payload(
        self,
        payload: dict[str, Any],
        request: PredictionRequest,
        now: datetime,
    ) -> StructuredFootballFeature | None:
        impact_area = str(payload.get("impact_area") or "")
        if impact_area not in {"lineup_availability", "tactical_matchup", "sentiment_narrative"}:
            return None
        try:
            direction = max(min(float(payload.get("direction", 0)), 1), -1)
            magnitude = max(min(float(payload.get("magnitude", 0)), 0.55), 0)
            confidence = max(min(float(payload.get("confidence", 0)), 0.58), 0)
        except (TypeError, ValueError):
            return None
        if confidence <= 0 or magnitude <= 0 or abs(direction) < 0.05:
            return None
        return StructuredFootballFeature(
            feature_type=str(payload.get("feature_type") or "unofficial_signal"),
            impact_area=impact_area,
            feature_value={
                "supporting_indices": payload.get("supporting_indices") or [],
                "unofficial": True,
            },
            direction=1.0 if direction > 0 else -1.0,
            magnitude=magnitude,
            confidence=confidence,
            feature_confidence=min(confidence * 0.9, 0.55),
            snapshot_at=now,
            available_at=now,
            prediction_deadline=request.prediction_deadline,
            leakage_risk="medium",
            extraction_method="llm_unofficial_signal_structuring",
            source_name=self.source_name,
            rationale=str(payload.get("rationale") or "AI structured unofficial candidate signal."),
        )

    def _supporting_claims(self, indices: Any, candidates: list[dict[str, Any]]) -> list[str]:
        if not isinstance(indices, list):
            return []
        claims: list[str] = []
        by_index = {item["index"]: item for item in candidates}
        for raw in indices[:4]:
            try:
                item = by_index.get(int(raw))
            except (TypeError, ValueError):
                item = None
            if item:
                claims.append(str(item["text"])[:180])
        return claims

    def _gap(self, message: str) -> EvidenceItem:
        return EvidenceItem(
            claim=message,
            source=self.source_name,
            source_query="AI unofficial signal structuring",
            evidence_stage="collection_gap",
            impact_area="sentiment_narrative",
            source_reliability=0.35,
            recency_score=0.0,
            confidence=0.0,
        )
