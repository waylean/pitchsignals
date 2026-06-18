from __future__ import annotations

import json
from typing import Any

from app.core.llm import get_llm_client
from app.domain_packs.base import DomainPack
from app.schemas import PredictionRequest


class LLMQuestionDecomposer:
    """Use the configured LLM to normalize user questions before collection.

    This step is intentionally limited to task understanding: canonical team
    names, search terms, and event context. It must not create evidence or
    probabilities.
    """

    async def decompose(self, request: PredictionRequest, pack: DomainPack) -> PredictionRequest:
        if request.context.get("skip_ai_decomposition"):
            return request

        client = get_llm_client()
        system = (
            "You decompose prediction questions for a strict evidence workflow. "
            "Do not predict the outcome. Do not invent match facts. "
            "Return canonical English names for teams/countries when possible. "
            "If the question asks who wins/will win, normalize it as a future prediction "
            "unless it explicitly asks for an already finished result. "
            "If uncertain, keep confidence low and include alternatives. "
            "Return JSON only."
        )
        user = json.dumps(
            {
                "domain_pack": pack.key,
                "question": request.question,
                "existing_context": request.context,
                "expected_output": {
                    "status": "ok|partial|skipped",
                    "normalized_question": "short canonical question",
                    "competitors": ["canonical first side", "canonical second side"],
                    "competition": "competition or null",
                    "event_time_hint": "date/time text or null",
                    "search_query": "compact English search query",
                    "language": "zh|en|other",
                    "confidence": "0..1",
                    "notes": ["string"],
                },
            },
            ensure_ascii=False,
        )
        try:
            result = await client.complete_json(system, user)
        except Exception as exc:
            request.context["ai_decomposition"] = {"status": "error", "reason": repr(exc)}
            return request

        if not isinstance(result, dict):
            return request
        request.context["ai_decomposition"] = result

        competitors = result.get("competitors")
        if (
            isinstance(competitors, list)
            and len(competitors) >= 2
            and all(isinstance(item, str) and item.strip() for item in competitors[:2])
        ):
            request.context.setdefault("competitors", [competitors[0].strip(), competitors[1].strip()])

        for key in ["normalized_question", "competition", "event_time_hint", "search_query", "language"]:
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                if key == "normalized_question":
                    value = self._future_safe_question(request.question, value)
                if key == "search_query":
                    value = self._future_safe_query(request.question, value)
                request.context.setdefault(key, value.strip())

        confidence = result.get("confidence")
        if isinstance(confidence, (int, float)):
            request.context.setdefault("decomposition_confidence", max(min(float(confidence), 1), 0))

        return request

    def _future_safe_question(self, original: str, normalized: str) -> str:
        lowered = normalized.lower()
        asks_winner = "\u8c01\u8d62" in original or "who wins" in original.lower() or "who will win" in original.lower()
        explicit_past = any(token in original for token in ["\u5df2\u7ecf", "\u7ed3\u679c", "\u7ed3\u675f"]) or any(
            token in original.lower() for token in ["already", "result", "finished", "ended"]
        )
        if asks_winner and not explicit_past and "who won" in lowered:
            return normalized.replace("Who won", "Who will win").replace("who won", "who will win")
        return normalized

    def _future_safe_query(self, original: str, query: str) -> str:
        asks_winner = "\u8c01\u8d62" in original or "who wins" in original.lower() or "who will win" in original.lower()
        explicit_past = any(token in original for token in ["\u5df2\u7ecf", "\u7ed3\u679c", "\u7ed3\u675f"]) or any(
            token in original.lower() for token in ["already", "result", "finished", "ended"]
        )
        if asks_winner and not explicit_past and " result" not in query.lower():
            return f"{query} preview prediction"
        return query
