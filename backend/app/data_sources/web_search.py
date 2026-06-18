from __future__ import annotations

import asyncio
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote, unquote, urlparse

import httpx

from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] | None = None
        self._capture: str | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        classes = set(attr.get("class", "").split())

        if tag == "a" and "result__a" in classes:
            self._current = {"title": "", "url": self._clean_url(attr.get("href", "")), "snippet": ""}
            self._capture = "title"
            self._parts = []
        elif self._current is not None and tag in {"a", "div"} and "result__snippet" in classes:
            self._capture = "snippet"
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def handle_endtag(self, tag: str) -> None:
        if self._current is None or self._capture is None:
            return
        if self._capture == "title" and tag == "a":
            self._current["title"] = unescape(" ".join(self._parts))
            self._capture = None
            self._parts = []
        elif self._capture == "snippet" and tag in {"a", "div"}:
            self._current["snippet"] = unescape(" ".join(self._parts))
            self.results.append(self._current)
            self._current = None
            self._capture = None
            self._parts = []

    def close(self) -> None:
        super().close()
        if self._current and self._current.get("title"):
            self.results.append(self._current)
            self._current = None

    def _clean_url(self, href: str) -> str:
        if not href:
            return ""
        parsed = urlparse(href)
        if parsed.path.startswith("/l/"):
            target = parse_qs(parsed.query).get("uddg", [""])[0]
            return unquote(target)
        return href


class WebSearchSource:
    """Zero-key web search collector.

    DuckDuckGo lightweight HTML results are first-pass candidates, not verified
    facts. Queries and hosts that look like rumor/fan sources are explicitly
    downgraded while remaining usable for aggregate direction.
    """

    name = "web_search"

    def __init__(self, max_queries: int = 5, results_per_query: int = 3):
        self.max_queries = max_queries
        self.results_per_query = results_per_query

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        evidence: list[EvidenceItem] = []
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0 ForecastIntelligence/0.1"},
        ) as client:
            batches = await asyncio.gather(
                *(self._search_one(client, query) for query in queries[: self.max_queries])
            )
            for batch in batches:
                evidence.extend(batch)
        return evidence

    async def _search_one(self, client: httpx.AsyncClient, query: EvidenceQuery) -> list[EvidenceItem]:
        url = f"https://html.duckduckgo.com/html/?q={quote(query.text)}"
        try:
            response = await client.get(url)
            response.raise_for_status()
            response.encoding = "utf-8"
        except Exception as exc:
            return [
                EvidenceItem(
                    claim=f"Search failed for {query.factor_label or query.impact_area}: {exc!r}",
                    source=self.name,
                    source_url=url,
                    source_query=query.text,
                    evidence_stage="collection_error",
                    collected_at=datetime.utcnow(),
                    impact_area="collection_error",
                    source_reliability=0.35,
                    recency_score=0.5,
                    confidence=0.2,
                )
            ]

        parser = _DuckDuckGoResultParser()
        parser.feed(response.text)
        parser.close()

        items: list[EvidenceItem] = []
        for result in parser.results[: self.results_per_query]:
            title = result.get("title", "").strip()
            snippet = result.get("snippet", "").strip()
            result_url = result.get("url", "").strip()
            if not title:
                continue

            unofficial = self._is_unofficial_query(query.text) or self._is_unofficial_url(result_url)
            claim = f"{title}"
            if snippet:
                claim = f"{title} - {snippet}"
            claim = self._clean_text(claim)

            reliability = self._source_reliability(result_url)
            confidence = 0.52 if snippet else 0.42
            if unofficial:
                reliability = min(reliability, 0.46)
                confidence = min(confidence, 0.42)

            items.append(
                EvidenceItem(
                    claim=claim[:900],
                    source=self.name,
                    source_url=result_url or url,
                    source_query=query.text,
                    evidence_stage="candidate",
                    collected_at=datetime.utcnow(),
                    impact_area=query.impact_area,
                    source_reliability=reliability,
                    recency_score=self._recency_from_text(claim),
                    corroboration_count=0,
                    contradiction_count=0,
                    confidence=confidence,
                    verifier_notes=(
                        ["Unofficial/local-media/fan evidence candidate; aggregate before using directionally."]
                        if unofficial
                        else []
                    ),
                )
            )
        return items

    def _source_reliability(self, url: str) -> float:
        host = urlparse(url).netloc.lower()
        if any(key in host for key in ["fifa.com", "uefa.com", "nba.com", "sec.gov", "imf.org"]):
            return 0.9
        if any(key in host for key in ["reuters.com", "apnews.com", "espn.com", "bbc.", "theathletic.com"]):
            return 0.78
        if any(key in host for key in ["wikipedia.org", "worldfootball.net", "transfermarkt."]):
            return 0.65
        if any(key in host for key in ["reddit.com", "x.com", "twitter.com"]):
            return 0.42
        return 0.55

    def _is_unofficial_query(self, query: str) -> bool:
        lowered = query.lower()
        return any(
            token in lowered
            for token in [
                "rumor",
                "rumour",
                "leaked",
                "fan forum",
                "reddit",
                "local media",
                "predicted xi",
                "fan tactical",
            ]
        )

    def _is_unofficial_url(self, url: str) -> bool:
        host = urlparse(url).netloc.lower()
        return any(key in host for key in ["reddit.com", "x.com", "twitter.com", "forum", "fans"])

    def _recency_from_text(self, text: str) -> float:
        lowered = text.lower()
        if any(token in lowered for token in ["2026", "today", "yesterday", "latest", "updated"]):
            return 0.82
        if any(token in lowered for token in ["2025", "2024"]):
            return 0.58
        return 0.45

    def _clean_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        replacements = {
            "\u2013": "-",
            "\u2014": "-",
            "\u2018": "'",
            "\u2019": "'",
            "\u201c": '"',
            "\u201d": '"',
        }
        for bad, good in replacements.items():
            cleaned = cleaned.replace(bad, good)
        return cleaned
