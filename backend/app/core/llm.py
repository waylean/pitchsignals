import json
import re
from typing import Any, Protocol

import httpx

from app.core.config import settings


class LLMClient(Protocol):
    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Return structured JSON only."""


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat client.

    The LLM is intentionally used for structured reasoning tasks only. It should
    never be the direct authority for final probabilities.
    """

    def __init__(self, base_url: str | None, api_key: str | None, model: str):
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key
        self.model = model

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.api_key or not settings.llm_enabled:
            return {"status": "skipped", "reason": "LLM_API_KEY not configured"}

        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = self._parse_json_content(content)
        parsed.setdefault("status", "ok")
        parsed["_llm"] = {"provider": settings.llm_provider, "model": self.model}
        return parsed

    def _parse_json_content(self, content: str) -> dict[str, Any]:
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {"status": "error", "reason": "LLM returned non-object JSON"}
        except json.JSONDecodeError:
            pass

        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                return parsed if isinstance(parsed, dict) else {"status": "error", "reason": "LLM returned non-object JSON"}
            except json.JSONDecodeError:
                pass

        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            candidate = content[start : end + 1]
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {"status": "error", "reason": "LLM returned non-object JSON"}
            except json.JSONDecodeError:
                pass
        return {
            "status": "ok",
            "summary": content[:1200] or "LLM review completed, but returned empty content.",
            "parse_warning": "LLM returned non-JSON content; summary was preserved as plain text.",
        }


def get_llm_client() -> LLMClient:
    return OpenAICompatibleClient(settings.llm_base_url, settings.llm_api_key, settings.llm_model)
