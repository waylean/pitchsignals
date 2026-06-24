from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import httpx

from app.core.config import settings
from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest


class ZhihuSearchSource:
    """Optional Zhihu Open Platform search adapter.

    The adapter is disabled by default. It is useful for Chinese-language
    football context and the Zhihu Vibe Coding activity, while the zero-key
    public search path remains the default fallback.
    """

    name = "zhihu_search"
    _base_url = "https://developer.zhihu.com/api/v1/content"

    def __init__(self):
        self.enabled = settings.zhihu_search_enabled
        self.access_secret = settings.zhihu_access_secret
        self.mode = settings.zhihu_search_mode
        self.max_queries = max(settings.zhihu_search_max_queries, 0)
        self.results_per_query = min(max(settings.zhihu_search_results_per_query, 1), 10)

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if not self.enabled or not self.access_secret:
            return []

        selected = queries[: self.max_queries]
        if not selected:
            return []

        async with httpx.AsyncClient(
            timeout=20,
            headers={
                "Authorization": f"Bearer {self.access_secret}",
                "X-Request-Timestamp": str(int(datetime.utcnow().timestamp())),
                "Content-Type": "application/json",
                "User-Agent": "PitchSignals/1.0",
            },
        ) as client:
            batches = await asyncio.gather(*(self._search_one(client, query) for query in selected))

        evidence: list[EvidenceItem] = []
        for batch in batches:
            evidence.extend(batch)
        return evidence

    async def _search_one(self, client: httpx.AsyncClient, query: EvidenceQuery) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        for endpoint in self._endpoints():
            try:
                response = await client.get(
                    f"{self._base_url}/{endpoint}",
                    params={"Query": query.text, "Count": self.results_per_query},
                )
                response.raise_for_status()
                payload = response.json()
            except Exception as exc:
                evidence.append(self._error(query, endpoint, repr(exc)))
                continue

            code = payload.get("Code")
            if code not in {0, 200}:
                evidence.append(self._error(query, endpoint, str(payload.get("Message") or code)))
                continue

            data = payload.get("Data") or {}
            items = data.get("Items") or []
            for raw in items[: self.results_per_query]:
                item = self._item_from_raw(raw, query, endpoint)
                if item:
                    evidence.append(item)
        return evidence

    def _endpoints(self) -> list[str]:
        normalized = self.mode.lower().strip()
        if normalized == "global":
            return ["global_search"]
        if normalized == "both":
            return ["zhihu_search", "global_search"]
        return ["zhihu_search"]

    def _item_from_raw(self, raw: dict[str, Any], query: EvidenceQuery, endpoint: str) -> EvidenceItem | None:
        title = self._clean(str(raw.get("Title") or raw.get("title") or ""))
        content = self._clean(str(raw.get("Content") or raw.get("content") or ""))
        url = str(raw.get("Url") or raw.get("URL") or raw.get("url") or "")
        if not title and not content:
            return None

        claim = f"{title} - {content}" if title and content else title or content
        published_at = self._published_at(raw)
        source_name = "zhihu_search" if endpoint == "zhihu_search" else "zhihu_global_search"
        reliability = 0.62 if endpoint == "zhihu_search" else 0.56

        return EvidenceItem(
            claim=claim[:900],
            source=source_name,
            source_url=url or None,
            license_note="Collected through Zhihu Open Platform API; follow Zhihu platform terms when reusing content.",
            source_provenance={
                "provider": "zhihu_open_platform",
                "endpoint": endpoint,
                "vote_up_count": raw.get("VoteUpCount"),
                "comment_count": raw.get("CommentCount"),
            },
            source_query=query.text,
            evidence_stage="candidate",
            raw_excerpt=content[:500] if content else None,
            published_at=published_at,
            collected_at=datetime.utcnow(),
            impact_area=query.impact_area,
            source_reliability=reliability,
            recency_score=self._recency(published_at, claim),
            corroboration_count=0,
            contradiction_count=0,
            confidence=0.5 if content else 0.42,
            verifier_notes=[
                "Zhihu Open Platform search result; verify timestamp and source context before using directionally."
            ],
        )

    def _error(self, query: EvidenceQuery, endpoint: str, message: str) -> EvidenceItem:
        return EvidenceItem(
            claim=f"Zhihu Open Platform {endpoint} collection failed: {message}",
            source=self.name,
            source_query=query.text,
            evidence_stage="collection_error",
            collected_at=datetime.utcnow(),
            impact_area="collection_error",
            source_reliability=0.25,
            recency_score=0.0,
            confidence=0.0,
        )

    def _published_at(self, raw: dict[str, Any]) -> datetime | None:
        keys = ["CreatedTime", "UpdatedTime", "PublishedTime", "created_time", "updated_time", "published_time"]
        for key in keys:
            value = raw.get(key)
            if isinstance(value, (int, float)) and value > 0:
                try:
                    return datetime.fromtimestamp(value)
                except (OSError, OverflowError, ValueError):
                    return None
        return None

    def _recency(self, published_at: datetime | None, text: str) -> float:
        if published_at:
            age_days = max((datetime.utcnow() - published_at).days, 0)
            if age_days <= 7:
                return 0.88
            if age_days <= 30:
                return 0.72
            if age_days <= 180:
                return 0.55
            return 0.35
        lowered = text.lower()
        if any(token in lowered for token in ["2026", "今天", "昨日", "最新", "world cup 2026"]):
            return 0.74
        return 0.45

    def _clean(self, text: str) -> str:
        return " ".join(text.split())
