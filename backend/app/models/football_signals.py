from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.entities import TEAM_ALIASES
from app.evidence.scoring import evidence_quality
from app.schemas import EvidenceItem, PredictionRequest, StructuredFootballFeature


@dataclass(frozen=True)
class FootballSignal:
    factor_key: str
    value: float
    confidence: float
    description: str


_FAVORITE_PATTERNS = [
    re.compile(r"\b(?P<team>[a-z][a-z\s.'-]{1,40}?)\s+(?:are|is|as)?\s*(?:the\s+)?favou?rites?\b", re.I),
    re.compile(r"\bbookmakers\s+place\s+(?P<team>[a-z][a-z\s.'-]{1,40}?)\s+as\s+favou?rites?\b", re.I),
    re.compile(r"\b(?P<team>[a-z][a-z\s.'-]{1,40}?)\s+(?:to\s+win|win)\s+at\s+@?\s*(?P<odds>\d+(?:\.\d+)?)\b", re.I),
]

_ODDS_NEAR_TEAM = re.compile(
    r"\b(?P<team>[a-z][a-z\s.'-]{1,40}?)\b.{0,45}?(?:@|odds?\s+of\s+)?(?P<odds>\d+(?:\.\d+)?)\b",
    re.I,
)

_RANKING_NEAR_TEAM = re.compile(
    r"\b(?P<team>[a-z][a-z\s.'-]{1,40}?)\b.{0,35}?\b(?:rank(?:ed|ing)?|#)\s*(?P<rank>\d{1,3})\b",
    re.I,
)

_IMPLIED_1X2 = re.compile(
    r"probabilities\s+home=(?P<home>0?\.\d+|1(?:\.0+)?)"
    r".{0,24}?draw=(?P<draw>0?\.\d+|1(?:\.0+)?)"
    r".{0,24}?away=(?P<away>0?\.\d+|1(?:\.0+)?)",
    re.I,
)

_RATING_FOR_TEAM_TEMPLATE = r"\b{team}\b.{{0,36}}?\brating\s*(?P<rating>\d{{3,4}})\b"
_GROUP_FORM_TEMPLATE = (
    r"\b{team}\b.{{0,30}}\bpoints\s*(?P<points>-?\d+)"
    r".{{0,30}}\bgoal\s*difference\s*(?P<gd>[+-]?\d+)"
)
_SPI_FOR_TEAM_TEMPLATE = r"\b{team}\b.{{0,36}}?\bspi\s*(?P<spi>\d{{1,3}}(?:\.\d+)?)\b"

_CHINESE_POSITIVE_TERMS = [
    "\u5360\u4f18",
    "\u4f18\u52bf",
    "\u66f4\u5f3a",
    "\u72b6\u6001\u66f4\u597d",
    "\u72b6\u6001\u51fa\u8272",
    "\u9635\u5bb9\u6df1\u5ea6",
    "\u5b9e\u529b\u66f4\u5f3a",
    "\u70ed\u95e8",
    "\u770b\u597d",
    "\u6709\u671b",
    "\u80dc\u7b97",
    "\u9886\u5148",
    "\u538b\u5236",
    "\u5168\u9762\u5360\u4f18",
]

_CHINESE_NEGATIVE_TERMS = [
    "\u7f3a\u9635",
    "\u4f24\u505c",
    "\u53d7\u4f24",
    "\u505c\u8d5b",
    "\u7f3a\u4e4f\u6bd4\u8d5b\u72b6\u6001",
    "\u72b6\u6001\u4e0d\u8db3",
    "\u72b6\u6001\u4f4e\u8ff7",
    "\u4f53\u80fd\u4e0d\u8db3",
    "\u75b2\u52b3",
    "\u4e0d\u5229",
    "\u9632\u7ebf\u6f0f\u6d1e",
    "\u77db\u76fe",
    "\u5185\u8ba7",
]

_ENGLISH_POSITIVE_TERMS = [
    "advantage",
    "favored",
    "favourite",
    "favorite",
    "strong",
    "stronger",
    "in form",
    "boost",
    "fit",
    "available",
    "returns",
    "full strength",
    "expected to start",
    "likely to start",
    "trained",
    "rested",
    "fresh",
    "dangerous",
    "counter attack",
    "counterattack",
    "set piece threat",
    "pace",
    "dominant",
]

_ENGLISH_NEGATIVE_TERMS = [
    "injury",
    "injured",
    "doubt",
    "doubtful",
    "suspended",
    "suspension",
    "misses",
    "missing",
    "without",
    "absence",
    "questionable",
    "late fitness test",
    "not expected to start",
    "benched",
    "rotation risk",
    "unavailable",
    "fatigue",
    "weakness",
    "vulnerable",
    "struggle",
    "struggles",
    "problem",
    "slow",
    "leaky",
]


class FootballSignalExtractor:
    def extract(
        self,
        evidence: list[EvidenceItem],
        competitors: tuple[str | None, str | None],
        request: PredictionRequest | None = None,
    ) -> list[FootballSignal]:
        first, second = competitors
        if not first or not second:
            return []
        first = first.lower()
        second = second.lower()
        signals: list[FootballSignal] = []
        for item in evidence:
            signals.extend(self._structured_feature_signals(item, request))
            text = self._canonical_text(self._text(item))
            quality = evidence_quality(item)
            if item.impact_area == "market_odds":
                signals.extend(self._market_signals(text, item, first, second, quality))
            if item.impact_area == "team_strength":
                signals.extend(self._ranking_signals(text, item, first, second, quality))
            signals.extend(self._narrative_direction_signals(text, item, first, second, quality))
        return signals

    def _structured_feature_signals(
        self,
        item: EvidenceItem,
        request: PredictionRequest | None,
    ) -> list[FootballSignal]:
        signals: list[FootballSignal] = []
        allowed_types = set()
        if request:
            allowed_types = set(request.context.get("allowed_feature_types") or [])
        for feature in item.structured_features:
            if not self._feature_allowed(feature, allowed_types):
                continue
            confidence = feature.feature_confidence
            if confidence is None:
                confidence = feature.confidence
            confidence = max(min(confidence, 1), 0)
            value = max(min(feature.direction * feature.magnitude, 1), -1)
            if confidence <= 0 or abs(value) < 0.001:
                continue
            signals.append(
                FootballSignal(
                    factor_key=feature.impact_area,
                    value=value,
                    confidence=confidence,
                    description=feature.rationale
                    or f"Structured {feature.feature_type} feature implies a directional edge.",
                )
            )
        return signals

    def _feature_allowed(
        self,
        feature: StructuredFootballFeature,
        allowed_types: set[str],
    ) -> bool:
        if not feature.horizon_allowed:
            return False
        if allowed_types and feature.feature_type not in allowed_types:
            return False
        if feature.prediction_deadline and feature.available_at:
            try:
                return feature.available_at <= feature.prediction_deadline
            except TypeError:
                return feature.available_at.replace(tzinfo=None) <= feature.prediction_deadline.replace(tzinfo=None)
        return True

    def _market_signals(
        self,
        text: str,
        item: EvidenceItem,
        first: str,
        second: str,
        quality: float,
    ) -> list[FootballSignal]:
        signals: list[FootballSignal] = []
        lower = text.lower()
        for pattern in _FAVORITE_PATTERNS:
            for match in pattern.finditer(lower):
                team = self._normalize_team(match.group("team"), first, second)
                if not team:
                    continue
                direction = 1 if team == first else -1
                confidence = min(0.55 + quality * 0.35, 0.9)
                signals.append(
                    FootballSignal(
                        factor_key=item.impact_area,
                        value=direction * 0.72,
                        confidence=confidence,
                        description=f"Market text identifies {team} as favourite.",
                    )
                )

        odds_by_team: dict[str, float] = {}
        implied = _IMPLIED_1X2.search(lower)
        if implied:
            home = float(implied.group("home"))
            away = float(implied.group("away"))
            signals.append(
                FootballSignal(
                    factor_key=item.impact_area,
                    value=max(min((home - away) / max(home + away, 0.001), 1), -1),
                    confidence=min(0.62 + quality * 0.28, 0.9),
                    description="Overround-adjusted 1X2 odds imply a directional market edge.",
                )
            )

        for match in _ODDS_NEAR_TEAM.finditer(lower):
            team = self._normalize_team(match.group("team"), first, second)
            if not team:
                continue
            odds = float(match.group("odds"))
            if 1.01 <= odds <= 20:
                odds_by_team[team] = min(odds_by_team.get(team, odds), odds)

        if first in odds_by_team and second in odds_by_team:
            first_prob = 1 / odds_by_team[first]
            second_prob = 1 / odds_by_team[second]
            total = first_prob + second_prob
            if total > 0:
                value = (first_prob - second_prob) / total
                signals.append(
                    FootballSignal(
                        factor_key=item.impact_area,
                        value=max(min(value, 1), -1),
                        confidence=min(0.5 + quality * 0.35, 0.85),
                        description="Decimal odds imply a directional market edge.",
                    )
                )
        return signals

    def _ranking_signals(
        self,
        text: str,
        item: EvidenceItem,
        first: str,
        second: str,
        quality: float,
    ) -> list[FootballSignal]:
        ranks: dict[str, int] = {}
        lower = text.lower()
        for team in (first, second):
            rank = self._rank_for_team(lower, team)
            if rank is not None:
                ranks[team] = rank
        if first in ranks and second in ranks:
            # Lower rank number is better. Squash difference into [-1, 1].
            diff = ranks[second] - ranks[first]
            value = max(min(diff / 60, 1), -1)
            return [
                FootballSignal(
                    factor_key=item.impact_area,
                    value=value,
                    confidence=min(0.45 + quality * 0.35, 0.8),
                    description="Ranking text implies a team-strength edge.",
                )
            ]
        ratings: dict[str, int] = {}
        for team in (first, second):
            rating = self._number_for_team(lower, team, _RATING_FOR_TEAM_TEMPLATE, "rating")
            if rating is not None:
                ratings[team] = rating
        if first in ratings and second in ratings:
            diff = ratings[first] - ratings[second]
            value = max(min(diff / 400, 1), -1)
            return [
                FootballSignal(
                    factor_key=item.impact_area,
                    value=value,
                    confidence=min(0.48 + quality * 0.35, 0.82),
                    description="Historical Elo rating text implies a team-strength edge.",
                )
            ]
        spis: dict[str, float] = {}
        for team in (first, second):
            spi = self._float_for_team(lower, team, _SPI_FOR_TEAM_TEMPLATE, "spi")
            if spi is not None:
                spis[team] = spi
        if first in spis and second in spis:
            diff = spis[first] - spis[second]
            return [
                FootballSignal(
                    factor_key=item.impact_area,
                    value=max(min(diff / 45, 1), -1),
                    confidence=min(0.5 + quality * 0.34, 0.82),
                    description="FiveThirtyEight SPI text implies a team-strength edge.",
                )
            ]
        form = self._group_form_signal(lower, item, first, second, quality)
        if form:
            return [form]
        return []

    def _narrative_direction_signals(
        self,
        text: str,
        item: EvidenceItem,
        first: str,
        second: str,
        quality: float,
    ) -> list[FootballSignal]:
        if item.evidence_stage in {"collection_error", "collection_gap", "excluded_after_deadline"}:
            return []
        signals: list[FootballSignal] = []
        lower = text.lower()
        confidence = min(0.36 + quality * 0.28, 0.66)
        for team in (first, second):
            opponents = [candidate for candidate in (first, second) if candidate != team]
            positive_hits = self._team_term_hits(lower, team, _positive_terms(item.impact_area), opponents)
            negative_hits = self._team_term_hits(lower, team, _negative_terms(item.impact_area), opponents)
            if positive_hits:
                direction = 1 if team == first else -1
                signals.append(
                    FootballSignal(
                        factor_key=item.impact_area,
                        value=direction * min(0.26 + 0.08 * positive_hits, 0.5),
                        confidence=confidence,
                        description=f"Narrative text contains positive match signals for {team}.",
                    )
                )
            if negative_hits:
                direction = -1 if team == first else 1
                signals.append(
                    FootballSignal(
                        factor_key=item.impact_area,
                        value=direction * min(0.22 + 0.07 * negative_hits, 0.46),
                        confidence=confidence,
                        description=f"Narrative text contains negative availability or form signals for {team}.",
                    )
                )
        return signals

    def _team_term_hits(self, text: str, team: str, terms: list[str], opponents: list[str]) -> int:
        escaped = re.escape(team)
        hits = 0
        for match in re.finditer(rf"\b{escaped}\b", text, re.I):
            window = text[match.end() : match.end() + 70]
            window = self._first_clause(window)
            before_window = self._last_clause(text[max(0, match.start() - 70) : match.start()])
            for opponent in opponents:
                opponent_index = window.find(opponent)
                if opponent_index >= 0:
                    window = window[:opponent_index]
                before_opponent_index = before_window.rfind(opponent)
                if before_opponent_index >= 0:
                    before_window = ""
            for term in terms:
                if term in window or term in before_window:
                    hits += 1
        return hits

    def _first_clause(self, text: str) -> str:
        end = len(text)
        separators = [".", "!", "?", ";", ",", "\n", "\u3002", "\uff01", "\uff1f", "\uff1b", "\uff0c"]
        for separator in separators:
            index = text.find(separator)
            if index >= 0:
                end = min(end, index)
        return text[:end]

    def _last_clause(self, text: str) -> str:
        start = 0
        separators = [".", "!", "?", ";", ",", "\n", "\u3002", "\uff01", "\uff1f", "\uff1b", "\uff0c"]
        for separator in separators:
            index = text.rfind(separator)
            if index >= 0:
                start = max(start, index + len(separator))
        return text[start:]

    def _rank_for_team(self, text: str, team: str) -> int | None:
        escaped = re.escape(team)
        patterns = [
            re.compile(rf"\b{escaped}\b.{{0,24}}\brank(?:ed|ing)?\s*(?P<rank>\d{{1,3}})\b", re.I),
            re.compile(rf"\b{escaped}\b.{{0,24}}#\s*(?P<rank>\d{{1,3}})\b", re.I),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return int(match.group("rank"))
        return None

    def _number_for_team(self, text: str, team: str, template: str, field: str) -> int | None:
        pattern = re.compile(template.format(team=re.escape(team)), re.I)
        match = pattern.search(text)
        if not match:
            return None
        return int(match.group(field))

    def _float_for_team(self, text: str, team: str, template: str, field: str) -> float | None:
        pattern = re.compile(template.format(team=re.escape(team)), re.I)
        match = pattern.search(text)
        if not match:
            return None
        return float(match.group(field))

    def _group_form_signal(
        self,
        text: str,
        item: EvidenceItem,
        first: str,
        second: str,
        quality: float,
    ) -> FootballSignal | None:
        first_form = self._form_for_team(text, first)
        second_form = self._form_for_team(text, second)
        if not first_form or not second_form:
            return None
        point_diff = first_form[0] - second_form[0]
        gd_diff = first_form[1] - second_form[1]
        value = max(min(point_diff / 6 + gd_diff / 12, 1), -1)
        return FootballSignal(
            factor_key=item.impact_area,
            value=value,
            confidence=min(0.42 + quality * 0.3, 0.72),
            description="Group-stage form before kickoff implies a team-strength edge.",
        )

    def _form_for_team(self, text: str, team: str) -> tuple[int, int] | None:
        pattern = re.compile(_GROUP_FORM_TEMPLATE.format(team=re.escape(team)), re.I)
        match = pattern.search(text)
        if not match:
            return None
        return int(match.group("points")), int(match.group("gd"))

    def _normalize_team(self, team_text: str, first: str, second: str) -> str | None:
        team = re.sub(r"[^a-z\s.'-]", " ", team_text.lower())
        team = " ".join(team.split())
        if first in team or team.endswith(first):
            return first
        if second in team or team.endswith(second):
            return second
        return None

    def _text(self, item: EvidenceItem) -> str:
        return f"{item.claim}\n{item.raw_excerpt or ''}"

    def _canonical_text(self, text: str) -> str:
        canonical = text
        for source, target in TEAM_ALIASES.items():
            canonical = canonical.replace(source, f" {target} ")
        return canonical


def _positive_terms(impact_area: str) -> list[str]:
    terms = [*_CHINESE_POSITIVE_TERMS, *_ENGLISH_POSITIVE_TERMS]
    if impact_area == "lineup_availability":
        return [
            *terms,
            "key player returns",
            "squad boost",
            "available again",
            "expected xi boost",
            "strong lineup",
            "full squad",
        ]
    if impact_area == "tactical_matchup":
        return [
            *terms,
            "tactical edge",
            "matchup advantage",
            "space behind",
            "creates overloads",
            "pressing advantage",
            "transition threat",
            "set-piece advantage",
            "formation suits",
        ]
    return terms


def _negative_terms(impact_area: str) -> list[str]:
    terms = [*_CHINESE_NEGATIVE_TERMS, *_ENGLISH_NEGATIVE_TERMS]
    if impact_area == "lineup_availability":
        return [
            *terms,
            "ruled out",
            "late fitness test",
            "not expected to start",
            "weakened lineup",
            "thin squad",
            "major absence",
        ]
    if impact_area == "tactical_matchup":
        return [
            *terms,
            "tactical weakness",
            "exposed",
            "struggles against",
            "space in behind",
            "vulnerable to pressing",
            "vulnerable in transition",
            "set-piece weakness",
            "formation mismatch",
        ]
    return terms
