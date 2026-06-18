from __future__ import annotations

import math

from app.domain_packs.base import DomainPack
from app.models.football_expected_goals import estimate_expected_goals
from app.models.governance import PredictionDistribution
from app.schemas import EvidenceItem, FactorScore, PredictionRequest


class FootballPoissonModel:
    model_id = "football_poisson_score_grid"
    supported_domains = {"football"}

    def predict_distribution(
        self,
        request: PredictionRequest,
        pack: DomainPack,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
    ) -> PredictionDistribution | None:
        expected = estimate_expected_goals(request.context, evidence, factors)
        if not expected:
            return None
        first_xg, second_xg = expected.first_xg, expected.second_xg
        first_win, draw, second_win = poisson_1x2(first_xg, second_xg)
        outcomes = request.outcomes or ["home_win", "draw", "away_win"]
        if len(outcomes) < 3:
            return None
        return PredictionDistribution(
            outcomes={outcomes[0]: first_win, outcomes[1]: draw, outcomes[2]: second_win},
            model_id=self.model_id,
            model_version="0.1",
            rationale=(
                f"Poisson score grid from {expected.source} "
                f"{first_xg:.2f}-{second_xg:.2f}."
            ),
        )


class FootballDixonColesModel:
    model_id = "football_dixon_coles_score_grid"
    supported_domains = {"football"}

    def predict_distribution(
        self,
        request: PredictionRequest,
        pack: DomainPack,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
    ) -> PredictionDistribution | None:
        expected = estimate_expected_goals(request.context, evidence, factors)
        if not expected:
            return None
        first_xg, second_xg = expected.first_xg, expected.second_xg
        raw = request.context.get("football_math") or {}
        rho = raw.get("dixon_coles_rho", raw.get("rho", -0.08)) if isinstance(raw, dict) else -0.08
        try:
            rho = max(min(float(rho), 0.2), -0.2)
        except (TypeError, ValueError):
            rho = -0.08
        first_win, draw, second_win = dixon_coles_1x2(first_xg, second_xg, rho)
        outcomes = request.outcomes or ["home_win", "draw", "away_win"]
        if len(outcomes) < 3:
            return None
        return PredictionDistribution(
            outcomes={outcomes[0]: first_win, outcomes[1]: draw, outcomes[2]: second_win},
            model_id=self.model_id,
            model_version="0.1",
            rationale=(
                f"Dixon-Coles score grid from {expected.source} {first_xg:.2f}-{second_xg:.2f} "
                f"with rho={rho:.2f}."
            ),
        )


class FootballMarketModel:
    model_id = "football_market_only"
    supported_domains = {"football"}

    def predict_distribution(
        self,
        request: PredictionRequest,
        pack: DomainPack,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
    ) -> PredictionDistribution | None:
        best = None
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
        outcomes = request.outcomes or ["home_win", "draw", "away_win"]
        if len(outcomes) < 3:
            return None
        return PredictionDistribution(
            outcomes={
                outcomes[0]: float(best["first_prob"]),
                outcomes[1]: float(best["draw_prob"]),
                outcomes[2]: float(best["second_prob"]),
            },
            model_id=self.model_id,
            model_version="0.1",
            rationale="Market-only baseline from overround-adjusted structured 1X2 odds.",
        )


class FootballEloStrengthModel:
    model_id = "football_elo_strength_baseline"
    supported_domains = {"football"}

    def predict_distribution(
        self,
        request: PredictionRequest,
        pack: DomainPack,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
    ) -> PredictionDistribution | None:
        strength = next((factor for factor in factors if factor.key == "team_strength"), None)
        if not strength or abs(strength.value) < 0.01 or strength.confidence <= 0:
            return None
        outcomes = request.outcomes or ["home_win", "draw", "away_win"]
        if len(outcomes) < 3:
            return None
        draw = max(min(0.28 + (1 - abs(strength.value)) * 0.08, 0.34), 0.22)
        decisive = 1 - draw
        first = decisive * (0.5 + max(min(strength.value, 1), -1) * 0.28)
        second = decisive - first
        return PredictionDistribution(
            outcomes={outcomes[0]: first, outcomes[1]: draw, outcomes[2]: second},
            model_id=self.model_id,
            model_version="0.1",
            rationale=(
                f"Elo/team-strength baseline from factor value {strength.value:.2f} "
                f"with confidence {strength.confidence:.0%}."
            ),
        )


def poisson_1x2(first_xg: float, second_xg: float, max_goals: int = 10) -> tuple[float, float, float]:
    return _score_grid_1x2(first_xg, second_xg, max_goals=max_goals)


def dixon_coles_1x2(
    first_xg: float,
    second_xg: float,
    rho: float = -0.08,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    return _score_grid_1x2(first_xg, second_xg, rho=rho, max_goals=max_goals)


def _score_grid_1x2(
    first_xg: float,
    second_xg: float,
    rho: float | None = None,
    max_goals: int = 10,
) -> tuple[float, float, float]:
    first_probs = [_poisson_pmf(goals, first_xg) for goals in range(max_goals + 1)]
    second_probs = [_poisson_pmf(goals, second_xg) for goals in range(max_goals + 1)]
    first_win = 0.0
    draw = 0.0
    second_win = 0.0
    for first_goals, first_prob in enumerate(first_probs):
        for second_goals, second_prob in enumerate(second_probs):
            tau = (
                _dixon_coles_tau(first_goals, second_goals, first_xg, second_xg, rho)
                if rho is not None
                else 1.0
            )
            prob = max(first_prob * second_prob * tau, 0)
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


def _dixon_coles_tau(
    first_goals: int,
    second_goals: int,
    first_xg: float,
    second_xg: float,
    rho: float,
) -> float:
    if first_goals == 0 and second_goals == 0:
        return 1 - first_xg * second_xg * rho
    if first_goals == 0 and second_goals == 1:
        return 1 + first_xg * rho
    if first_goals == 1 and second_goals == 0:
        return 1 + second_xg * rho
    if first_goals == 1 and second_goals == 1:
        return 1 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)
