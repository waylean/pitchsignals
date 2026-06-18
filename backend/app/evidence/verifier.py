from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from app.schemas import EvidenceItem


_DATE_PATTERNS = [
    re.compile(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b"),
    re.compile(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
        r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
        r"([0-3]?\d),?\s+(20\d{2})\b",
        re.IGNORECASE,
    ),
]

_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


class EvidenceVerifier:
    """Clean, dedupe, enrich, and conservatively verify evidence candidates."""

    def __init__(self, max_enrich: int = 8):
        self.max_enrich = max_enrich

    async def verify(
        self,
        evidence: list[EvidenceItem],
        prediction_deadline: datetime | None = None,
    ) -> list[EvidenceItem]:
        deduped = self._dedupe(evidence)
        await self._enrich_raw(deduped)
        for item in deduped:
            self._finalize_item(item, prediction_deadline)
        return deduped

    def _dedupe(self, evidence: list[EvidenceItem]) -> list[EvidenceItem]:
        merged: dict[str, EvidenceItem] = {}
        for item in evidence:
            key = self._dedupe_key(item)
            item.dedupe_key = key
            if key not in merged:
                merged[key] = item
                continue
            existing = merged[key]
            existing.corroboration_count += 1
            existing.confidence = min(max(existing.confidence, item.confidence) + 0.04, 0.95)
            existing.verifier_notes.append(f"Duplicate candidate merged from {item.source}.")
        return list(merged.values())

    async def _enrich_raw(self, evidence: list[EvidenceItem]) -> None:
        candidates = [
            item
            for item in evidence
            if item.source_url
            and item.evidence_stage == "candidate"
            and item.impact_area != "collection_error"
        ][: self.max_enrich]
        async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "ForecastIntelligence/0.1"}) as client:
            await asyncio.gather(*(self._fetch_reader_excerpt(client, item) for item in candidates))

    async def _fetch_reader_excerpt(self, client: httpx.AsyncClient, item: EvidenceItem) -> None:
        if not item.source_url:
            return
        reader_url = f"https://r.jina.ai/{item.source_url}"
        try:
            response = await client.get(reader_url)
            response.raise_for_status()
            response.encoding = "utf-8"
        except Exception as exc:
            item.verifier_notes.append(f"Raw page enrichment failed: {exc}")
            return

        text = self._clean_text(response.text)
        if not text:
            return
        item.raw_excerpt = text[:1200]
        parsed_date = self._parse_date(text[:2500])
        if parsed_date and not item.published_at:
            item.published_at = parsed_date
        item.confidence = min(item.confidence + 0.08, 0.95)
        item.verifier_notes.append("Raw page excerpt fetched through public Jina Reader.")

    def _finalize_item(self, item: EvidenceItem, prediction_deadline: datetime | None) -> None:
        item.evidence_id = item.evidence_id or self._evidence_id(item)
        if prediction_deadline and item.published_at:
            deadline = prediction_deadline
            published = item.published_at
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            item.deadline_compliant = published <= deadline
            if not item.deadline_compliant:
                item.evidence_stage = "excluded_after_deadline"
                item.verifier_notes.append("Excluded from trusted interpretation: published after deadline.")

        if item.evidence_stage == "candidate":
            if item.raw_excerpt and item.source_reliability >= 0.7:
                item.evidence_stage = "verified_candidate"
            elif item.raw_excerpt:
                item.evidence_stage = "enriched_candidate"
            else:
                item.evidence_stage = "candidate"

    def _dedupe_key(self, item: EvidenceItem) -> str:
        host = urlparse(item.source_url or "").netloc.lower()
        claim = re.sub(r"\W+", " ", item.claim.lower()).strip()[:180]
        return hashlib.sha1(f"{host}|{claim}|{item.impact_area}".encode("utf-8")).hexdigest()

    def _evidence_id(self, item: EvidenceItem) -> str:
        return hashlib.sha1(
            f"{item.dedupe_key}|{item.source}|{item.collected_at.isoformat()}".encode("utf-8")
        ).hexdigest()[:16]

    def _parse_date(self, text: str) -> datetime | None:
        match = _DATE_PATTERNS[0].search(text)
        if match:
            year, month, day = (int(part) for part in match.groups())
            return datetime(year, month, day, tzinfo=timezone.utc)

        match = _DATE_PATTERNS[1].search(text)
        if match:
            month_text, day, year = match.groups()
            month = _MONTHS[month_text[:3].lower()]
            return datetime(int(year), month, int(day), tzinfo=timezone.utc)
        return None

    def _clean_text(self, text: str) -> str:
        cleaned = " ".join(text.split())
        replacements = {
            "â": "—",
            "â€“": "–",
            "â": "'",
            "â": '"',
            "â": '"',
            "бк": "—",
        }
        for bad, good in replacements.items():
            cleaned = cleaned.replace(bad, good)
        return cleaned
