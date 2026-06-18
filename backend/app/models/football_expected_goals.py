from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas import EvidenceItem, FactorScore


@dataclass(frozen=True)
class ExpectedGoalsEstimate:
    first_xg: float
    second_xg: float
    source: str
    confidence: float


def estimate_expected_goals(
    context: dict[str, Any],
    evidence: list[EvidenceItem],
    factors: list[FactorScore],
) -> ExpectedGoalsEstimate | None:
    supplied = _supplied_expected_goals(context)
    if supplied:
        first, second = supplied
        return ExpectedGoalsEstimate(first, second, "supplied_expected_goals", 0.92)

    market = _market_expected_goals(evidence)
    strength = _team_strength(factors)
    if market and strength is not None:
        return ExpectedGoalsEstimate(
            _clamp_xg(market.first_xg + strength * 0.08),
            _clamp_xg(market.second_xg - strength * 0.08),
            "market_implied_xg_with_strength_adjustment",
            min(market.confidence + 0.03, 0.86),
        )
    if market:
        return market
    if strength is not None:
        total_goals = 2.45
        edge = strength * 0.9
        return ExpectedGoalsEstimate(
            _clamp_xg(total_goals / 2 + edge / 2),
            _clamp_xg(total_goals / 2 - edge / 2),
            "team_strength_implied_xg",
            0.58,
        )
    return None


def _supplied_expected_goals(context: dict[str, Any]) -> tuple[float, float] | None:
    raw = context.get("football_math") or {}
    if isinstance(raw, dict):
        expected = raw.get("expected_goals") or raw.get("xg")
    else:
        expected = None
    expected = expected or context.get("expected_goals") or context.get("xg")

    if isinstance(expected, dict):
        first = expected.get("first") or expected.get("home") or expected.get("team1")
        second = expected.get("second") or expected.get("away") or expected.get("team2")
    elif isinstance(expected, (list, tuple)) and len(expected) >= 2:
        first, second = expected[0], expected[1]
    else:
        return None
    try:
        return _clamp_xg(float(first)), _clamp_xg(float(second))
    except (TypeError, ValueError):
        return None


def _market_expected_goals(evidence: list[EvidenceItem]) -> ExpectedGoalsEstimate | None:
    best: dict[str, float] | None = None
    best_confidence = -1.0
    for item in evidence:
        for feature in item.structured_features:
            if feature.feature_type != "odds" or feature.impact_area != "market_odds":
                continue
            confidence = (
                feature.feature_confidence
                if feature.feature_confidence is not None
                else feature.confidence
            )
            if confidence > best_confidence:
                best = feature.feature_value
                best_confidence = confidence
    if not best:
        return None
    try:
        first_prob = float(best["first_prob"])
        draw_prob = float(best["draw_prob"])
        second_prob = float(best["second_prob"])
    except (KeyError, TypeError, ValueError):
        return None

    first_xg, second_xg = _fit_xg_to_1x2(first_prob, draw_prob, second_prob)
    return ExpectedGoalsEstimate(first_xg, second_xg, "market_implied_xg", min(best_confidence, 0.83))


def _team_strength(factors: list[FactorScore]) -> float | None:
    factor = next((item for item in factors if item.key == "team_strength"), None)
    if not factor or factor.confidence <= 0 or abs(factor.value) < 0.05:
        return None
    return max(min(factor.value, 1), -1)


def _fit_xg_to_1x2(first_prob: float, draw_prob: float, second_prob: float) -> tuple[float, float]:
    target_total = first_prob + draw_prob + second_prob
    if target_total <= 0:
        return 1.25, 1.25
    target = (
        first_prob / target_total,
        draw_prob / target_total,
        second_prob / target_total,
    )
    best_error = float("inf")
    best_pair = (1.25, 1.25)
    for first_step in range(5, 81):
        first_xg = first_step * 0.05
        for second_step in range(5, 81):
            second_xg = second_step * 0.05
            implied = _poisson_1x2(first_xg, second_xg)
            error = (
                (implied[0] - target[0]) ** 2
                + 1.35 * (implied[1] - target[1]) ** 2
                + (implied[2] - target[2]) ** 2
                + 0.01 * max(first_xg + second_xg - 3.8, 0) ** 2
            )
            if error < best_error:
                best_error = error
                best_pair = (first_xg, second_xg)
    return best_pair


def _poisson_1x2(
    first_xg: float,
    second_xg: float,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    first_probs = [_poisson_pmf(goals, first_xg) for goals in range(max_goals + 1)]
    second_probs = [_poisson_pmf(goals, second_xg) for goals in range(max_goals + 1)]
    first_win = draw = second_win = 0.0
    for first_goals, first_score_prob in enumerate(first_probs):
        for second_goals, second_score_prob in enumerate(second_probs):
            prob = first_score_prob * second_score_prob
            if first_goals > second_goals:
                first_win += prob
            elif first_goals == second_goals:
                draw += prob
            else:
                second_win += prob
    total = first_win + draw + second_win
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return first_win / total, draw / total, second_win / total


def _poisson_pmf(k: int, lam: float) -> float:
    import math

    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _clamp_xg(value: float) -> float:
    return max(min(float(value), 6), 0.05)
