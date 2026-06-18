from app.core.entities import infer_competitors
from app.domain_packs.base import DomainPack
from app.evidence.scoring import evidence_quality
from app.models.football_signals import FootballSignal, FootballSignalExtractor
from app.schemas import EvidenceItem, FactorScore, PredictionRequest


_POSITIVE_TERMS = {
    "win",
    "wins",
    "winner",
    "favourite",
    "favorite",
    "favored",
    "advantage",
    "strong",
    "stronger",
    "best bet",
    "qualify",
    "higher",
    "ranked",
}

_NEGATIVE_TERMS = {
    "lose",
    "loses",
    "underdog",
    "injury",
    "injured",
    "suspended",
    "absence",
    "doubt",
    "weaker",
}


class WeightedPredictionModel:
    """Transparent baseline model.

    Factors are normalized into a directional signal in [-1, 1]. This remains
    transparent and conservative: weak evidence changes confidence more than
    probability, and absent directional evidence leaves a neutral baseline.
    """

    def __init__(self):
        self.football_signals = FootballSignalExtractor()

    def score_factors(
        self,
        pack: DomainPack,
        request: PredictionRequest,
        evidence: list[EvidenceItem],
    ) -> list[FactorScore]:
        weights = self._weight_map(pack, request)
        scores: list[FactorScore] = []
        competitors = infer_competitors(request)
        structured_signals = (
            self.football_signals.extract(evidence, competitors, request)
            if pack.key == "football"
            else []
        )

        for factor in pack.factors:
            relevant = [
                item
                for item in evidence
                if item.impact_area == factor.key
                and item.evidence_stage
                not in {"collection_error", "collection_gap", "excluded_after_deadline"}
            ]
            evidence_count = len(relevant)
            qualities = [evidence_quality(item) for item in relevant]
            confidence = round(sum(qualities) / len(qualities), 4) if qualities else 0.0
            factor_signals = [
                signal for signal in structured_signals if signal.factor_key == factor.key
            ]
            if factor_signals:
                value = self._weighted_signal_value(factor_signals)
                signal_confidence = sum(signal.confidence for signal in factor_signals) / len(
                    factor_signals
                )
                confidence = round(max(confidence, signal_confidence), 4)
                rationale = self._signal_rationale(value, confidence, factor_signals, competitors)
            elif pack.key == "football" and factor.key == "market_odds":
                value = 0.0
                rationale = "Market evidence collected, but no match-specific structured odds were extracted."
            elif pack.key == "football" and factor.key == "lineup_availability":
                proxy = self._consensus_proxy(scores, ["market_odds", "team_strength"])
                if abs(proxy) >= 0.05:
                    value = max(min(proxy * 0.45, 1), -1)
                    confidence = round(max(confidence, 0.36), 4)
                    leader = competitors[0] if value > 0 else competitors[1]
                    rationale = (
                        "No confirmed lineup or injury signal was structured, so a conservative "
                        f"squad-depth and availability proxy weakly favors {leader}; "
                        f"factor confidence is {confidence:.0%}."
                    )
                else:
                    value = 0.0
                    rationale = self._rationale(value, confidence, evidence_count, competitors)
            elif pack.key == "football" and factor.key == "tactical_matchup":
                proxy = self._consensus_proxy(scores, ["market_odds", "team_strength"])
                if abs(proxy) >= 0.05:
                    value = max(min(proxy * 0.55, 1), -1)
                    confidence = round(max(confidence, 0.42), 4)
                    leader = competitors[0] if value > 0 else competitors[1]
                    rationale = (
                        "No explicit tactical matchup signal was structured, so a conservative "
                        f"market-and-strength tactical proxy weakly favors {leader}; "
                        f"factor confidence is {confidence:.0%}."
                    )
                else:
                    value = 0.0
                    rationale = self._rationale(value, confidence, evidence_count, competitors)
            else:
                value = self._directional_value(relevant, competitors)
                rationale = self._rationale(value, confidence, evidence_count, competitors)

            scores.append(
                FactorScore(
                    key=factor.key,
                    label=factor.label,
                    value=value,
                    weight=weights[factor.key],
                    confidence=confidence,
                    evidence_count=evidence_count,
                    rationale=rationale,
                )
            )
        return scores

    def predict(self, request: PredictionRequest, factors: list[FactorScore]) -> dict[str, float]:
        outcomes = request.outcomes or ["yes", "no"]
        if len(outcomes) == 1:
            return {outcomes[0]: 1.0}

        directional_signal = sum(f.value * f.weight * f.confidence for f in factors)
        base = 1 / len(outcomes)
        probabilities = {outcome: base for outcome in outcomes}

        if len(outcomes) >= 2:
            scale = 0.36 if len(outcomes) == 3 and request.domain == "football" else 0.18
            limit = 0.26 if len(outcomes) == 3 and request.domain == "football" else 0.14
            shift = max(min(directional_signal * scale, limit), -limit)
            probabilities[outcomes[0]] = max(0.01, base + shift)
            probabilities[outcomes[-1]] = max(0.01, base - shift)
            if len(outcomes) == 3 and request.domain == "football":
                probabilities[outcomes[1]] = max(0.12, base - abs(shift) * 0.42)

        total = sum(probabilities.values())
        return {key: round(value / total, 4) for key, value in probabilities.items()}

    def _weight_map(self, pack: DomainPack, request: PredictionRequest) -> dict[str, float]:
        configured = pack.factor_weight_map()
        profile = request.context.get("weight_profile")
        if not isinstance(profile, dict):
            return configured

        raw_weights: dict[str, float] = {}
        for factor in pack.factors:
            value = profile.get(factor.key, configured[factor.key])
            try:
                raw_weights[factor.key] = max(float(value), 0)
            except (TypeError, ValueError):
                raw_weights[factor.key] = configured[factor.key]

        total = sum(raw_weights.values()) or 1
        return {key: value / total for key, value in raw_weights.items()}

    def has_directional_signal(self, factors: list[FactorScore]) -> bool:
        return any(abs(f.value) >= 0.05 and f.confidence > 0 for f in factors)

    def _weighted_signal_value(self, signals: list[FootballSignal]) -> float:
        total_confidence = sum(signal.confidence for signal in signals)
        if total_confidence <= 0:
            return 0.0
        value = sum(signal.value * signal.confidence for signal in signals) / total_confidence
        return max(min(value, 1), -1)

    def _consensus_proxy(self, scores: list[FactorScore], keys: list[str]) -> float:
        selected = [score for score in scores if score.key in keys and score.confidence > 0]
        total_weight = sum(score.weight * score.confidence for score in selected)
        if total_weight <= 0:
            return 0.0
        value = sum(score.value * score.weight * score.confidence for score in selected) / total_weight
        return max(min(value, 1), -1)

    def _directional_value(
        self,
        evidence: list[EvidenceItem],
        competitors: tuple[str | None, str | None],
    ) -> float:
        first, second = competitors
        if not first or not second:
            return 0.0

        first_score = 0.0
        second_score = 0.0
        for item in evidence:
            text = item.claim.lower()
            quality = evidence_quality(item)
            first_mentions = text.count(first)
            second_mentions = text.count(second)
            positivity = sum(1 for term in _POSITIVE_TERMS if term in text)
            negativity = sum(1 for term in _NEGATIVE_TERMS if term in text)
            directional_terms = max(positivity - negativity * 0.6, 0)
            if directional_terms <= 0:
                continue
            if first_mentions > second_mentions:
                first_score += quality * directional_terms
            elif second_mentions > first_mentions:
                second_score += quality * directional_terms

        total = first_score + second_score
        if total <= 0:
            return 0.0
        return max(min((first_score - second_score) / total, 1), -1)

    def _rationale(
        self,
        value: float,
        confidence: float,
        evidence_count: int,
        competitors: tuple[str | None, str | None],
    ) -> str:
        if evidence_count == 0:
            return "No factor-specific evidence collected yet."
        if not competitors[0] or not competitors[1]:
            return "Evidence collected, but competitors were not reliably extracted."
        if abs(value) < 0.05:
            return "Evidence collected, but no reliable directional signal was extracted."
        leader = competitors[0] if value > 0 else competitors[1]
        return f"Candidate evidence weakly favors {leader}; factor confidence is {confidence:.0%}."

    def _signal_rationale(
        self,
        value: float,
        confidence: float,
        signals: list[FootballSignal],
        competitors: tuple[str | None, str | None],
    ) -> str:
        if abs(value) < 0.05:
            return "Structured signals were extracted, but they do not create a clear edge."
        leader = competitors[0] if value > 0 else competitors[1]
        source = signals[0].description if signals else "Structured signal extracted."
        return f"{source} Overall this weakly favors {leader}; factor confidence is {confidence:.0%}."
