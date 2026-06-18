from __future__ import annotations

import json
from typing import Any

from app.core.llm import get_llm_client
from app.domain_packs.base import DomainPack
from app.schemas import EvidenceItem, FactorScore, PredictionRequest


class LLMResearchReviewer:
    """LLM-assisted workflow reviewer.

    The reviewer audits evidence coverage and gaps after deterministic scoring.
    It must not change final probabilities.
    """

    async def review(
        self,
        request: PredictionRequest,
        pack: DomainPack,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
        outcomes: dict[str, float],
    ) -> dict[str, Any]:
        client = get_llm_client()
        system = (
            "You are a strict research auditor for an evidence-driven prediction workflow. "
            "Do not invent facts. Do not change probabilities. "
            "Assess whether each workflow route has enough evidence, which data is missing, "
            "and whether the final probability should remain conservative. "
            "For lineup availability and tactical matchup, candidate local-media, fan, and rumor evidence "
            "can count as partial support when it is clearly marked as unofficial and aggregated. "
            "If the user's question is written in Chinese, return Chinese strings in summary, "
            "route reasons, missing_data, risk_flags, and recommended_next_queries. "
            "Return JSON only."
        )
        user = json.dumps(
            {
                "domain_pack": pack.key,
                "question": request.question,
                "question_decomposition": request.context.get("ai_decomposition"),
                "outcomes": outcomes,
                "factors": [factor.model_dump() for factor in factors],
                "evidence_summary": self._evidence_summary(evidence),
                "required_routes": [
                    "data_layer",
                    "market_odds",
                    "team_strength",
                    "lineup_availability",
                    "tactical_matchup",
                    "referee_environment",
                    "sentiment_narrative",
                    "unofficial_signal_aggregation",
                    "prediction_weights",
                    "prediction",
                    "feedback_readiness",
                ],
                "output_schema": {
                    "status": "ok|partial|skipped|error",
                    "can_complete_all_routes": "boolean",
                    "summary": "short string",
                    "route_check": {
                        "route_name": {
                            "status": "complete|partial|missing",
                            "reason": "short string",
                        }
                    },
                    "missing_data": ["string"],
                    "risk_flags": ["string"],
                    "recommended_next_queries": ["string"],
                },
            },
            ensure_ascii=False,
        )
        try:
            result = await client.complete_json(system, user)
        except Exception as exc:
            return {"status": "error", "reason": f"LLM review failed: {exc}"}
        return self._normalize_review_result(result, request, evidence, factors)

    def _normalize_review_result(
        self,
        result: dict[str, Any],
        request: PredictionRequest,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
    ) -> dict[str, Any]:
        deterministic = self._deterministic_route_check(request, evidence, factors)
        route_check = result.get("route_check")
        if not isinstance(route_check, dict) or not route_check:
            result["route_check"] = deterministic["route_check"]
        else:
            for key, value in deterministic["route_check"].items():
                route_check.setdefault(key, value)

        summary = result.get("summary")
        if not isinstance(summary, str) or self._looks_like_raw_json(summary):
            result["summary"] = deterministic["summary"]
        result.setdefault("status", deterministic["status"])
        if result.get("status") == "ok" and deterministic["status"] == "partial":
            result["status"] = "partial"
        result.setdefault("can_complete_all_routes", deterministic["can_complete_all_routes"])
        result.setdefault("missing_data", deterministic["missing_data"])
        result.setdefault("risk_flags", deterministic["risk_flags"])
        result.setdefault("recommended_next_queries", deterministic["recommended_next_queries"])
        return result

    def _deterministic_route_check(
        self,
        request: PredictionRequest,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
    ) -> dict[str, Any]:
        zh = self._is_chinese(request.question)
        by_factor = {factor.key: factor for factor in factors}
        route_check: dict[str, dict[str, str]] = {}
        missing_data: list[str] = []
        recommended: list[str] = []

        total_usable = sum(1 for item in evidence if item.evidence_stage not in {"collection_gap", "collection_error"})
        route_check["data_layer"] = self._route(
            "complete" if total_usable >= 12 else "partial" if total_usable else "missing",
            f"{total_usable} usable evidence items collected." if not zh else f"已收集 {total_usable} 条可用证据。",
        )

        for key in ["market_odds", "team_strength", "lineup_availability", "tactical_matchup", "referee_environment", "sentiment_narrative"]:
            factor = by_factor.get(key)
            if not factor:
                continue
            status = "missing"
            if factor.evidence_count > 0 and factor.confidence >= 0.55:
                status = "complete"
            elif factor.evidence_count > 0 or (abs(factor.value) >= 0.05 and factor.confidence > 0):
                status = "partial"
            if status == "missing":
                missing_data.append(self._route_name(key, zh))
                recommended.append(self._recommended_query(key, request, zh))
            route_check[key] = self._route(status, factor.rationale)

        unofficial_count = sum(1 for item in evidence if any("Unofficial" in note for note in item.verifier_notes))
        route_check["unofficial_signal_aggregation"] = self._route(
            "partial" if unofficial_count else "missing",
            (
                f"{unofficial_count} unofficial/local-media/fan candidates were collected and downgraded."
                if not zh
                else f"已收集 {unofficial_count} 条非官方/本地媒体/球迷候选信息，并已降低可信度。"
            ),
        )
        if not unofficial_count:
            recommended.append(
                "\u641c\u7d22\u9884\u6d4b\u9996\u53d1\u3001\u672c\u5730\u5a92\u4f53\u3001\u7403\u8ff7\u6218\u672f\u5206\u6790\u548c\u4e34\u573a\u4f24\u505c\u4f20\u95fb\u3002"
                if zh
                else "Search predicted XI, local media, fan tactical analysis, and late availability rumors."
            )

        route_check["prediction_weights"] = self._route("complete", "\u5df2\u8fd0\u884c\u56e0\u7d20\u6743\u91cd\u548c\u8db3\u7403\u6570\u5b66\u96c6\u6210\u3002" if zh else "Factor weights and football ensemble models ran.")
        route_check["prediction"] = self._route("complete", "\u5df2\u751f\u6210\u4e09\u9879\u7ed3\u679c\u6982\u7387\u548c\u660e\u786e\u503e\u5411\u3002" if zh else "Three-way probabilities and a directional pick were generated.")
        route_check["feedback_readiness"] = self._route("complete", "\u9884\u6d4b\u53ef\u5199\u5165 ledger \u5e76\u652f\u6301\u8d5b\u540e\u56de\u586b\u3002" if zh else "Prediction can be logged and evaluated after the result.")

        status = "partial" if any(item["status"] in {"partial", "missing"} for item in route_check.values()) else "ok"
        summary = (
            "\u5df2\u5b8c\u6210 AI \u63d0\u95ee\u89e3\u6784\u3001\u8bc1\u636e\u91c7\u96c6\u3001\u56e0\u7d20\u8bc4\u5206\u548c\u6982\u7387\u751f\u6210\u3002\u9635\u5bb9\u548c\u6218\u672f\u4fe1\u606f\u53ef\u4f7f\u7528\u975e\u5b98\u65b9\u5019\u9009\u8bc1\u636e\u505a\u4f4e\u7f6e\u4fe1\u805a\u5408\uff0c\u4f46\u7f3a\u5c11\u786e\u8ba4\u9996\u53d1\u6216\u660e\u786e\u6218\u672f\u62a5\u544a\u65f6\uff0c\u7ed3\u8bba\u4ecd\u5e94\u4fdd\u5b88\u3002"
            if zh
            else "AI question decomposition, evidence collection, factor scoring, and probability generation completed. Lineup and tactical routes can use downgraded unofficial candidates, but conclusions remain conservative without confirmed lineup or explicit tactical reports."
        )
        return {
            "status": status,
            "can_complete_all_routes": status == "ok",
            "summary": summary,
            "route_check": route_check,
            "missing_data": missing_data[:8],
            "risk_flags": [
                "\u975e\u5b98\u65b9\u6d88\u606f\u53ea\u80fd\u4f5c\u4e3a\u4f4e\u7f6e\u4fe1\u5019\u9009\u8bc1\u636e\u3002" if zh else "Unofficial evidence is low-confidence candidate evidence only.",
            ],
            "recommended_next_queries": [item for item in recommended if item][:8],
        }

    def _route(self, status: str, reason: str) -> dict[str, str]:
        return {"status": status, "reason": reason}

    def _looks_like_raw_json(self, summary: str) -> bool:
        stripped = summary.strip()
        return stripped.startswith("{") or '"route_check"' in stripped or len(stripped) > 900

    def _is_chinese(self, text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _route_name(self, key: str, zh: bool) -> str:
        labels = {
            "market_odds": "\u5e02\u573a\u8d54\u7387",
            "team_strength": "\u7403\u961f\u5b9e\u529b",
            "lineup_availability": "\u9635\u5bb9\u4e0e\u4f24\u505c",
            "tactical_matchup": "\u6218\u672f\u5bf9\u4f4d",
            "referee_environment": "\u88c1\u5224\u4e0e\u73af\u5883",
            "sentiment_narrative": "\u8206\u8bba\u4e0e\u53d9\u4e8b",
        }
        return labels.get(key, key) if zh else key.replace("_", " ")

    def _recommended_query(self, key: str, request: PredictionRequest, zh: bool) -> str:
        base = str(request.context.get("search_query") or request.question)
        suffix = {
            "market_odds": "1X2 odds bookmaker h2h",
            "lineup_availability": "predicted XI injuries suspensions leaked lineup local media",
            "tactical_matchup": "tactical preview formation matchup pressing transition set pieces",
            "referee_environment": "referee appointment venue weather kickoff",
            "sentiment_narrative": "fan sentiment local media confidence morale",
        }.get(key, "")
        if not suffix:
            return ""
        return f"{base} {suffix}" if not zh else f"\u7ee7\u7eed\u641c\u7d22\uff1a{base} {suffix}"

    def _evidence_summary(self, evidence: list[EvidenceItem]) -> dict[str, Any]:
        by_area: dict[str, int] = {}
        by_source: dict[str, int] = {}
        gaps: list[str] = []
        examples: list[dict[str, str | None]] = []
        unofficial_count = 0
        for item in evidence:
            by_area[item.impact_area] = by_area.get(item.impact_area, 0) + 1
            by_source[item.source] = by_source.get(item.source, 0) + 1
            if any("Unofficial" in note for note in item.verifier_notes):
                unofficial_count += 1
            if item.evidence_stage == "collection_gap":
                gaps.append(item.claim)
            elif len(examples) < 10:
                examples.append(
                    {
                        "stage": item.evidence_stage,
                        "area": item.impact_area,
                        "source": item.source,
                        "claim": item.claim[:500],
                    }
                )
        return {
            "total": len(evidence),
            "by_area": by_area,
            "by_source": by_source,
            "unofficial_candidate_count": unofficial_count,
            "gaps": gaps[:12],
            "examples": examples,
        }
