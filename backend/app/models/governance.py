from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.domain_packs.base import DomainPack
from app.schemas import EvidenceItem, FactorScore, PredictionRequest


@dataclass(frozen=True)
class PredictionProblem:
    domain: str
    outcomes: list[str]
    question: str
    context: dict


@dataclass(frozen=True)
class PredictionDistribution:
    outcomes: dict[str, float]
    model_id: str
    model_version: str = "0.1"
    rationale: str | None = None

    def normalized(self) -> "PredictionDistribution":
        total = sum(max(value, 0) for value in self.outcomes.values()) or 1
        return PredictionDistribution(
            outcomes={key: round(max(value, 0) / total, 4) for key, value in self.outcomes.items()},
            model_id=self.model_id,
            model_version=self.model_version,
            rationale=self.rationale,
        )

    def entropy(self) -> float:
        values = [max(value, 1e-12) for value in self.normalized().outcomes.values()]
        return -sum(value * math.log(value) for value in values)


@dataclass(frozen=True)
class EvaluationCase:
    actual: str
    distribution: PredictionDistribution
    horizon_profile: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationReport:
    model_id: str
    case_count: int
    log_loss: float
    brier_score: float
    ranked_probability_score: float
    accuracy: float
    calibration_bins: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelLeaderboardEntry:
    model_id: str
    case_count: int
    log_loss: float
    brier_score: float
    ranked_probability_score: float
    accuracy: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelLeaderboard:
    leaderboard_id: str
    entries: list[ModelLeaderboardEntry]
    metric: str = "log_loss"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromotionGate:
    gate_id: str
    min_cases: int = 250
    max_log_loss_delta_vs_baseline: float = -0.01
    max_brier_delta_vs_baseline: float = -0.005
    max_calibration_error: float = 0.08
    min_calibration_bin_cases: int = 1
    required_slices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromotionDecision:
    gate_id: str
    candidate_model_id: str
    baseline_model_id: str
    decision: str
    reasons: list[str]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class MultiWindowPromotionGate:
    gate_id: str
    min_windows: int = 3
    min_total_test_cases: int = 5000
    require_zero_leakage: bool = True
    latest_window_must_pass: bool = True
    required_baselines: list[str] = field(
        default_factory=lambda: [
            "uniform_baseline",
            "home_prior_baseline",
            "football_data_market_only",
        ]
    )
    required_slice_prefixes: list[str] = field(
        default_factory=lambda: [
            "league:",
            "season:",
            "league_season:",
            "horizon:",
            "market_available:",
        ]
    )
    max_failed_required_slices: int = 0
    dedupe_slice_failures: bool = True


@dataclass(frozen=True)
class RollbackGate:
    gate_id: str
    min_cases: int = 250
    max_log_loss_delta_vs_baseline: float = 0.0
    max_brier_delta_vs_baseline: float = 0.0
    max_calibration_error: float = 0.12
    required_consecutive_failures: int = 2


@dataclass(frozen=True)
class RollbackDecision:
    gate_id: str
    stable_model_id: str
    baseline_model_id: str
    decision: str
    reasons: list[str]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ModelRun:
    model_id: str
    distribution: PredictionDistribution
    status: str = "ok"
    notes: list[str] | None = None


@dataclass(frozen=True)
class EnsembleProfile:
    profile_id: str
    version: str
    domain: str
    model_weights: dict[str, float]
    maturity: str = "experimental"
    horizon_profile: str | None = None
    competition_scope: str | None = None
    data_tier: str | None = None
    training_dataset_id: str | None = None
    validation_report_id: str | None = None
    test_report_id: str | None = None
    lineage: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    promoted_at: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def normalized_weights(self, available_model_ids: set[str]) -> dict[str, float]:
        weights = {
            model_id: max(float(weight), 0.0)
            for model_id, weight in self.model_weights.items()
            if model_id in available_model_ids
        }
        total = sum(weights.values())
        if total <= 0:
            return {}
        return {model_id: weight / total for model_id, weight in weights.items()}


@dataclass(frozen=True)
class ProfileLifecycleArtifact:
    artifact_id: str
    profile_id: str
    profile_version: str
    domain: str
    status: str
    decision: dict[str, Any]
    source_report_id: str | None = None
    created_at: str | None = None
    previous_profile_version: str | None = None
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    next_actions: list[str] = field(default_factory=list)


DEFAULT_ENSEMBLE_PROFILES: dict[str, EnsembleProfile] = {
    "football": EnsembleProfile(
        profile_id="football_research_alpha",
        version="0.6.0",
        domain="football",
        model_weights={
            "weighted_evidence": 0.70,
            "football_market_only": 0.12,
            "football_elo_strength_baseline": 0.06,
            "football_poisson_score_grid": 0.06,
            "football_dixon_coles_score_grid": 0.06,
        },
        maturity="research_alpha",
        notes=[
            "Default football ensemble profile before time-split calibration.",
            "Weights are conservative and should be promoted only through EvaluationReport gates.",
        ],
    )
}

PROFILE_STORE_ROOT = Path(__file__).resolve().parents[2] / "profiles"


def ensemble_profile_from_dict(raw: dict[str, Any], fallback_domain: str | None = None) -> EnsembleProfile:
    model_weights = raw.get("model_weights")
    if not isinstance(model_weights, dict) or not model_weights:
        raise ValueError("ensemble profile requires non-empty model_weights")
    domain = str(raw.get("domain") or fallback_domain or "")
    if not domain:
        raise ValueError("ensemble profile requires domain")
    return EnsembleProfile(
        profile_id=str(raw.get("profile_id") or f"{domain}_custom"),
        version=str(raw.get("version") or "custom"),
        domain=domain,
        model_weights={str(key): float(value) for key, value in model_weights.items()},
        maturity=str(raw.get("maturity") or "experimental"),
        horizon_profile=_optional_str(raw.get("horizon_profile")),
        competition_scope=_optional_str(raw.get("competition_scope")),
        data_tier=_optional_str(raw.get("data_tier")),
        training_dataset_id=_optional_str(raw.get("training_dataset_id")),
        validation_report_id=_optional_str(raw.get("validation_report_id")),
        test_report_id=_optional_str(raw.get("test_report_id")),
        lineage=raw.get("lineage") if isinstance(raw.get("lineage"), dict) else {},
        created_at=_optional_str(raw.get("created_at")),
        promoted_at=_optional_str(raw.get("promoted_at")),
        metrics=raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {},
        notes=[str(note) for note in raw.get("notes", [])] if isinstance(raw.get("notes"), list) else [],
    )


def load_ensemble_profiles(root: Path | None = None) -> dict[str, list[EnsembleProfile]]:
    root = root or PROFILE_STORE_ROOT
    profiles: dict[str, list[EnsembleProfile]] = {}
    if not root.exists():
        return profiles
    for path in sorted(root.rglob("*.json")):
        if ".lifecycle." in path.name or "lifecycle" in path.parts:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            profile = ensemble_profile_from_dict(raw)
        except Exception:
            continue
        profiles.setdefault(profile.domain, []).append(profile)
    return profiles


def profile_lifecycle_from_dict(raw: dict[str, Any]) -> ProfileLifecycleArtifact:
    profile_id = str(raw.get("profile_id") or "")
    domain = str(raw.get("domain") or "")
    if not profile_id or not domain:
        raise ValueError("profile lifecycle artifact requires profile_id and domain")
    return ProfileLifecycleArtifact(
        artifact_id=str(raw.get("artifact_id") or f"{profile_id}_lifecycle"),
        profile_id=profile_id,
        profile_version=str(raw.get("profile_version") or raw.get("version") or "unknown"),
        domain=domain,
        status=str(raw.get("status") or "candidate"),
        decision=raw.get("decision") if isinstance(raw.get("decision"), dict) else {},
        source_report_id=_optional_str(raw.get("source_report_id")),
        created_at=_optional_str(raw.get("created_at")),
        previous_profile_version=_optional_str(raw.get("previous_profile_version")),
        artifacts=(
            {str(key): str(value) for key, value in raw.get("artifacts", {}).items()}
            if isinstance(raw.get("artifacts"), dict)
            else {}
        ),
        metrics=raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {},
        next_actions=(
            [str(item) for item in raw.get("next_actions", [])]
            if isinstance(raw.get("next_actions"), list)
            else []
        ),
    )


def load_profile_lifecycle_artifacts(
    root: Path | None = None,
) -> dict[str, list[ProfileLifecycleArtifact]]:
    root = root or PROFILE_STORE_ROOT
    artifacts: dict[str, list[ProfileLifecycleArtifact]] = {}
    if not root.exists():
        return artifacts
    for path in sorted(root.rglob("*.lifecycle.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            artifact = profile_lifecycle_from_dict(raw)
        except Exception:
            continue
        artifacts.setdefault(artifact.domain, []).append(artifact)
    return artifacts


def get_default_ensemble_profile(
    domain: str,
    include_research_profiles: bool = True,
) -> EnsembleProfile | None:
    profiles = load_ensemble_profiles().get(domain, [])
    if profiles:
        stable = [profile for profile in profiles if profile.maturity in {"stable", "promoted"}]
        research = [
            profile
            for profile in profiles
            if include_research_profiles and profile.maturity in {"research_alpha", "experimental"}
        ]
        ordered = stable or research
        if not ordered:
            return DEFAULT_ENSEMBLE_PROFILES.get(domain) if include_research_profiles else None
        return sorted(ordered, key=lambda profile: profile.version, reverse=True)[0]
    return DEFAULT_ENSEMBLE_PROFILES.get(domain) if include_research_profiles else None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


class PredictionModel(Protocol):
    model_id: str
    supported_domains: set[str]

    def predict_distribution(
        self,
        request: PredictionRequest,
        pack: DomainPack,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
    ) -> PredictionDistribution | None:
        ...


class ModelRegistry:
    def __init__(self):
        self._models: list[PredictionModel] = []

    def register(self, model: PredictionModel) -> None:
        self._models.append(model)

    def run(
        self,
        request: PredictionRequest,
        pack: DomainPack,
        evidence: list[EvidenceItem],
        factors: list[FactorScore],
    ) -> list[ModelRun]:
        runs: list[ModelRun] = []
        for model in self._models:
            if pack.key not in model.supported_domains:
                continue
            try:
                distribution = model.predict_distribution(request, pack, evidence, factors)
            except Exception as exc:
                runs.append(
                    ModelRun(
                        model_id=model.model_id,
                        distribution=_neutral(request, model.model_id),
                        status="error",
                        notes=[str(exc)],
                    )
                )
                continue
            if distribution:
                runs.append(ModelRun(model_id=model.model_id, distribution=distribution.normalized()))
        return runs


def distribution_agreement(distributions: list[PredictionDistribution]) -> float:
    if len(distributions) <= 1:
        return 0.5
    normalized = [dist.normalized().outcomes for dist in distributions]
    keys = sorted(set().union(*(dist.keys() for dist in normalized)))
    avg_abs_deviation = 0.0
    count = 0
    for key in keys:
        values = [dist.get(key, 0.0) for dist in normalized]
        mean = sum(values) / len(values)
        avg_abs_deviation += sum(abs(value - mean) for value in values) / len(values)
        count += 1
    disagreement = avg_abs_deviation / max(count, 1)
    return round(max(min(1 - disagreement * 3, 1), 0), 4)


def select_ensemble_profile(domain: str, context: dict[str, Any]) -> EnsembleProfile | None:
    raw = context.get("ensemble_profile")
    if isinstance(raw, dict):
        profile = ensemble_profile_from_dict(raw, fallback_domain=domain)
        if context.get("horizon_profile") and not profile.horizon_profile:
            return EnsembleProfile(
                profile_id=profile.profile_id,
                version=profile.version,
                domain=profile.domain,
                model_weights=profile.model_weights,
                maturity=profile.maturity,
                horizon_profile=str(context.get("horizon_profile")),
                competition_scope=profile.competition_scope,
                data_tier=profile.data_tier,
                training_dataset_id=profile.training_dataset_id,
                validation_report_id=profile.validation_report_id,
                test_report_id=profile.test_report_id,
                lineage=profile.lineage,
                created_at=profile.created_at,
                promoted_at=profile.promoted_at,
                metrics=profile.metrics,
                notes=profile.notes,
            )
        return profile
    profile = get_default_ensemble_profile(domain)
    if profile and context.get("horizon_profile"):
        return EnsembleProfile(
            profile_id=profile.profile_id,
            version=profile.version,
            domain=profile.domain,
            model_weights=profile.model_weights,
            maturity=profile.maturity,
            horizon_profile=str(context.get("horizon_profile")),
            competition_scope=profile.competition_scope,
            data_tier=profile.data_tier,
            training_dataset_id=profile.training_dataset_id,
            validation_report_id=profile.validation_report_id,
            test_report_id=profile.test_report_id,
            lineage=profile.lineage,
            created_at=profile.created_at,
            promoted_at=profile.promoted_at,
            metrics=profile.metrics,
            notes=profile.notes,
        )
    return profile


def blend_distributions(
    primary: dict[str, float],
    auxiliaries: list[PredictionDistribution],
    auxiliary_weight: float = 0.25,
    ensemble_profile: EnsembleProfile | None = None,
) -> dict[str, float]:
    if ensemble_profile:
        primary_distribution = PredictionDistribution(
            outcomes=primary,
            model_id="weighted_evidence",
        ).normalized()
        by_model = {primary_distribution.model_id: primary_distribution}
        by_model.update({dist.model_id: dist.normalized() for dist in auxiliaries})
        weights = ensemble_profile.normalized_weights(set(by_model))
        if weights:
            keys = list(primary.keys())
            blended = {
                key: sum(
                    by_model[model_id].outcomes.get(key, 0.0) * weight
                    for model_id, weight in weights.items()
                )
                for key in keys
            }
            total = sum(blended.values()) or 1
            return {key: round(value / total, 4) for key, value in blended.items()}

    if not auxiliaries:
        return primary
    keys = list(primary.keys())
    aux_mean = {
        key: sum(dist.normalized().outcomes.get(key, 0.0) for dist in auxiliaries) / len(auxiliaries)
        for key in keys
    }
    blended = {
        key: primary[key] * (1 - auxiliary_weight) + aux_mean[key] * auxiliary_weight
        for key in keys
    }
    total = sum(blended.values()) or 1
    return {key: round(value / total, 4) for key, value in blended.items()}


def build_evaluation_report(
    model_id: str,
    cases: list[EvaluationCase],
    metadata: dict[str, Any] | None = None,
) -> EvaluationReport:
    if not cases:
        return EvaluationReport(
            model_id=model_id,
            case_count=0,
            log_loss=0.0,
            brier_score=0.0,
            ranked_probability_score=0.0,
            accuracy=0.0,
            calibration_bins=[],
            metadata=metadata or {},
        )

    log_loss_total = 0.0
    brier_total = 0.0
    rps_total = 0.0
    correct = 0
    confidence_rows: list[tuple[float, int]] = []
    for case in cases:
        outcomes = list(case.distribution.normalized().outcomes)
        probs = case.distribution.normalized().outcomes
        actual_probability = max(probs.get(case.actual, 0.0), 1e-9)
        log_loss_total += -math.log(actual_probability)
        brier_total += sum(
            (probs.get(outcome, 0.0) - (1.0 if outcome == case.actual else 0.0)) ** 2
            for outcome in outcomes
        )
        rps_total += ranked_probability_score(probs, case.actual, outcomes)
        top_outcome, top_probability = max(probs.items(), key=lambda item: item[1])
        is_correct = int(top_outcome == case.actual)
        correct += is_correct
        confidence_rows.append((top_probability, is_correct))

    count = len(cases)
    return EvaluationReport(
        model_id=model_id,
        case_count=count,
        log_loss=round(log_loss_total / count, 6),
        brier_score=round(brier_total / count, 6),
        ranked_probability_score=round(rps_total / count, 6),
        accuracy=round(correct / count, 6),
        calibration_bins=calibration_bins(confidence_rows),
        metadata=metadata or {},
    )


def ranked_probability_score(probs: dict[str, float], actual: str, outcomes: list[str]) -> float:
    if len(outcomes) <= 1:
        return 0.0
    cumulative_prediction = 0.0
    cumulative_actual = 0.0
    total = 0.0
    for outcome in outcomes[:-1]:
        cumulative_prediction += probs.get(outcome, 0.0)
        cumulative_actual += 1.0 if outcome == actual else 0.0
        total += (cumulative_prediction - cumulative_actual) ** 2
    return total / (len(outcomes) - 1)


def calibration_bins(rows: list[tuple[float, int]], bins: int = 5) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = []
        for confidence, correct in rows:
            if index == bins - 1:
                in_bucket = lower <= confidence <= upper
            else:
                in_bucket = lower <= confidence < upper
            if in_bucket:
                bucket.append((confidence, correct))
        if not bucket:
            results.append({"lower": lower, "upper": upper, "count": 0})
            continue
        avg_confidence = sum(item[0] for item in bucket) / len(bucket)
        accuracy = sum(item[1] for item in bucket) / len(bucket)
        results.append(
            {
                "lower": round(lower, 2),
                "upper": round(upper, 2),
                "count": len(bucket),
                "avg_confidence": round(avg_confidence, 4),
                "accuracy": round(accuracy, 4),
                "calibration_error": round(abs(avg_confidence - accuracy), 4),
            }
        )
    return results


def build_model_leaderboard(
    leaderboard_id: str,
    reports: list[EvaluationReport],
    metric: str = "log_loss",
    metadata: dict[str, Any] | None = None,
) -> ModelLeaderboard:
    ordered = sorted(reports, key=lambda report: getattr(report, metric))
    entries = [
        ModelLeaderboardEntry(
            model_id=report.model_id,
            case_count=report.case_count,
            log_loss=report.log_loss,
            brier_score=report.brier_score,
            ranked_probability_score=report.ranked_probability_score,
            accuracy=report.accuracy,
            rank=index + 1,
            metadata=report.metadata,
        )
        for index, report in enumerate(ordered)
    ]
    return ModelLeaderboard(
        leaderboard_id=leaderboard_id,
        entries=entries,
        metric=metric,
        metadata=metadata or {},
    )


def evaluate_promotion_gate(
    candidate: EvaluationReport,
    baseline: EvaluationReport,
    gate: PromotionGate | None = None,
    slice_reports: dict[str, tuple[EvaluationReport, EvaluationReport]] | None = None,
) -> PromotionDecision:
    gate = gate or PromotionGate(gate_id="default_football_1_0_gate")
    reasons: list[str] = []
    log_loss_delta = candidate.log_loss - baseline.log_loss
    brier_delta = candidate.brier_score - baseline.brier_score
    calibration = _calibration_summary(candidate, gate)
    max_calibration_error = calibration["max_supported_calibration_error"]

    if candidate.case_count < gate.min_cases:
        reasons.append(f"case_count {candidate.case_count} < required {gate.min_cases}")
    if log_loss_delta > gate.max_log_loss_delta_vs_baseline:
        reasons.append(
            f"log_loss_delta {log_loss_delta:.6f} is not better than threshold "
            f"{gate.max_log_loss_delta_vs_baseline:.6f}"
        )
    if brier_delta > gate.max_brier_delta_vs_baseline:
        reasons.append(
            f"brier_delta {brier_delta:.6f} is not better than threshold "
            f"{gate.max_brier_delta_vs_baseline:.6f}"
        )
    if max_calibration_error > gate.max_calibration_error:
        reasons.append(
            f"max_calibration_error {max_calibration_error:.6f} > allowed "
            f"{gate.max_calibration_error:.6f}"
        )

    slice_results: dict[str, Any] = {}
    if gate.required_slices:
        slice_reports = slice_reports or {}
        slice_gate = PromotionGate(
            gate_id=f"{gate.gate_id}_slice",
            min_cases=gate.min_cases,
            max_log_loss_delta_vs_baseline=gate.max_log_loss_delta_vs_baseline,
            max_brier_delta_vs_baseline=gate.max_brier_delta_vs_baseline,
            max_calibration_error=gate.max_calibration_error,
            min_calibration_bin_cases=gate.min_calibration_bin_cases,
            required_slices=[],
        )
        for slice_id in gate.required_slices:
            pair = slice_reports.get(slice_id)
            if pair is None:
                reasons.append(f"required slice {slice_id} is missing")
                slice_results[slice_id] = {"decision": "missing"}
                continue
            slice_candidate, slice_baseline = pair
            slice_decision = evaluate_promotion_gate(slice_candidate, slice_baseline, slice_gate)
            slice_results[slice_id] = {
                "decision": slice_decision.decision,
                "reasons": slice_decision.reasons,
                "metrics": slice_decision.metrics,
            }
            if slice_decision.decision != "promote":
                reasons.append(f"required slice {slice_id} did not pass promotion gate")

    decision = "promote" if not reasons else "hold"
    if decision == "promote":
        reasons.append("candidate passed all promotion thresholds")
    return PromotionDecision(
        gate_id=gate.gate_id,
        candidate_model_id=candidate.model_id,
        baseline_model_id=baseline.model_id,
        decision=decision,
        reasons=reasons,
        metrics={
            "candidate_cases": candidate.case_count,
            "baseline_cases": baseline.case_count,
            "log_loss_delta": round(log_loss_delta, 6),
            "brier_delta": round(brier_delta, 6),
            **calibration,
            "slice_results": slice_results,
        },
    )


def _calibration_summary(report: EvaluationReport, gate: PromotionGate) -> dict[str, Any]:
    supported_bins: list[dict[str, Any]] = []
    sparse_bins: list[dict[str, Any]] = []
    for bucket in report.calibration_bins:
        count = int(bucket.get("count") or 0)
        if count <= 0:
            continue
        item = {
            "lower": bucket.get("lower"),
            "upper": bucket.get("upper"),
            "count": count,
            "calibration_error": float(bucket.get("calibration_error", 0.0)),
        }
        if count >= gate.min_calibration_bin_cases:
            supported_bins.append(item)
        else:
            sparse_bins.append(item)
    all_bins = [*supported_bins, *sparse_bins]
    max_all = max((item["calibration_error"] for item in all_bins), default=0.0)
    max_supported = max((item["calibration_error"] for item in supported_bins), default=0.0)
    ece_denominator = sum(item["count"] for item in all_bins)
    ece = (
        sum(item["calibration_error"] * item["count"] for item in all_bins) / ece_denominator
        if ece_denominator
        else 0.0
    )
    return {
        "max_calibration_error": round(max_supported, 6),
        "max_supported_calibration_error": round(max_supported, 6),
        "max_calibration_error_all_bins": round(max_all, 6),
        "expected_calibration_error": round(ece, 6),
        "min_calibration_bin_cases": gate.min_calibration_bin_cases,
        "ignored_sparse_calibration_bins": sparse_bins,
    }


def evaluate_multi_window_promotion_gate(
    artifact: Any,
    gate: MultiWindowPromotionGate | None = None,
) -> PromotionDecision:
    gate = gate or MultiWindowPromotionGate(gate_id="default_multi_window_promotion_gate")
    artifact = _artifact_to_dict(artifact)
    windows = artifact.get("windows") if isinstance(artifact.get("windows"), list) else []
    summary = artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
    reasons: list[str] = []
    slice_failures: list[dict[str, Any]] = []
    canonical_slice_failures: dict[str, dict[str, Any]] = {}
    missing_slice_prefixes: dict[str, list[str]] = {}
    baseline_decisions: dict[str, list[str]] = {
        baseline: [] for baseline in gate.required_baselines
    }

    window_count = len(windows)
    total_test_cases = int(summary.get("test_cases") or 0)
    leakage_issues = int(summary.get("leakage_issues") or 0)
    if window_count < gate.min_windows:
        reasons.append(f"window_count {window_count} < required {gate.min_windows}")
    if total_test_cases < gate.min_total_test_cases:
        reasons.append(f"test_cases {total_test_cases} < required {gate.min_total_test_cases}")
    if gate.require_zero_leakage and leakage_issues:
        reasons.append(f"leakage_issues {leakage_issues} > 0")

    for window in windows:
        split = window.get("split", {})
        split_id = split.get("split_id", "unknown")
        baseline_map = {
            "uniform_baseline": window.get("promotion_decision_vs_uniform"),
            "home_prior_baseline": window.get("promotion_decision_vs_home_prior"),
            "football_data_market_only": window.get("promotion_decision_vs_raw_market"),
            "previous_stable_profile": window.get("promotion_decision_vs_previous_stable"),
        }
        for baseline in gate.required_baselines:
            decision = baseline_map.get(baseline)
            if not isinstance(decision, dict):
                baseline_decisions.setdefault(baseline, []).append("missing")
                reasons.append(f"{split_id} missing baseline decision for {baseline}")
                continue
            baseline_decisions.setdefault(baseline, []).append(str(decision.get("decision")))
            if decision.get("decision") != "promote":
                reasons.append(f"{split_id} failed baseline gate for {baseline}")

        raw_decision = window.get("promotion_decision_vs_raw_market")
        slice_results = {}
        if isinstance(raw_decision, dict):
            metrics = raw_decision.get("metrics") if isinstance(raw_decision.get("metrics"), dict) else {}
            slice_results = (
                metrics.get("slice_results") if isinstance(metrics.get("slice_results"), dict) else {}
            )
        for prefix in gate.required_slice_prefixes:
            if not any(str(slice_id).startswith(prefix) for slice_id in slice_results):
                missing_slice_prefixes.setdefault(split_id, []).append(prefix)
        for slice_id, slice_result in slice_results.items():
            if isinstance(slice_result, dict) and slice_result.get("decision") != "promote":
                slice_failures.append(
                    {
                        "split_id": split_id,
                        "slice_id": slice_id,
                        "decision": slice_result.get("decision"),
                        "reasons": slice_result.get("reasons", []),
                    }
                )
                canonical_key = _canonical_slice_failure_key(split_id, slice_id)
                canonical_slice_failures.setdefault(canonical_key, slice_failures[-1])

    for split_id, prefixes in missing_slice_prefixes.items():
        reasons.append(f"{split_id} missing required slice prefixes: {', '.join(prefixes)}")
    failed_slice_count = (
        len(canonical_slice_failures) if gate.dedupe_slice_failures else len(slice_failures)
    )
    if failed_slice_count > gate.max_failed_required_slices:
        reasons.append(
            f"failed_required_slices {failed_slice_count} > allowed {gate.max_failed_required_slices}"
        )
    latest_window = windows[-1] if windows else None
    if gate.latest_window_must_pass and isinstance(latest_window, dict):
        latest_raw = latest_window.get("promotion_decision_vs_raw_market")
        if not isinstance(latest_raw, dict) or latest_raw.get("decision") != "promote":
            latest_split = latest_window.get("split", {}).get("split_id", "latest")
            reasons.append(f"{latest_split} failed latest-window raw-market gate")

    decision = "promote" if not reasons else "hold"
    if decision == "promote":
        reasons.append("candidate passed multi-window promotion gate")
    return PromotionDecision(
        gate_id=gate.gate_id,
        candidate_model_id=str(artifact.get("candidate_model_id") or "football_data_market_calibrated"),
        baseline_model_id=",".join(gate.required_baselines),
        decision=decision,
        reasons=reasons,
        metrics={
            **summary,
            "baseline_decisions": baseline_decisions,
            "missing_slice_prefixes": missing_slice_prefixes,
            "failed_required_slices": slice_failures,
            "failed_required_slice_count": failed_slice_count,
            "canonical_failed_required_slices": list(canonical_slice_failures.values()),
        },
    )


def _canonical_slice_failure_key(split_id: str, slice_id: str) -> str:
    parts = slice_id.split(":")
    if len(parts) >= 3 and parts[0] == "league_season":
        return f"{split_id}:league:{parts[1]}"
    return f"{split_id}:{slice_id}"


def _artifact_to_dict(artifact: Any) -> dict[str, Any]:
    if isinstance(artifact, dict):
        return artifact
    to_dict = getattr(artifact, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return payload
    return {}


def evaluate_rollback_gate(
    stable_reports: list[EvaluationReport],
    baseline_reports: list[EvaluationReport],
    gate: RollbackGate | None = None,
) -> RollbackDecision:
    gate = gate or RollbackGate(gate_id="default_football_rollback_gate")
    if not stable_reports or not baseline_reports:
        return RollbackDecision(
            gate_id=gate.gate_id,
            stable_model_id=stable_reports[-1].model_id if stable_reports else "unknown",
            baseline_model_id=baseline_reports[-1].model_id if baseline_reports else "unknown",
            decision="watch",
            reasons=["rollback gate requires stable and baseline reports"],
            metrics={"windows": 0, "consecutive_failures": 0},
        )

    pairs = list(zip(stable_reports, baseline_reports))
    window_results: list[dict[str, Any]] = []
    consecutive_failures = 0
    for stable, baseline in pairs:
        max_calibration_error = max(
            (
                float(bucket.get("calibration_error", 0.0))
                for bucket in stable.calibration_bins
                if bucket.get("count", 0)
            ),
            default=0.0,
        )
        log_loss_delta = stable.log_loss - baseline.log_loss
        brier_delta = stable.brier_score - baseline.brier_score
        failed = False
        reasons: list[str] = []
        if stable.case_count < gate.min_cases:
            failed = True
            reasons.append(f"case_count {stable.case_count} < required {gate.min_cases}")
        if log_loss_delta > gate.max_log_loss_delta_vs_baseline:
            failed = True
            reasons.append(f"log_loss_delta {log_loss_delta:.6f} worse than allowed")
        if brier_delta > gate.max_brier_delta_vs_baseline:
            failed = True
            reasons.append(f"brier_delta {brier_delta:.6f} worse than allowed")
        if max_calibration_error > gate.max_calibration_error:
            failed = True
            reasons.append(f"max_calibration_error {max_calibration_error:.6f} > allowed")
        consecutive_failures = consecutive_failures + 1 if failed else 0
        window_results.append(
            {
                "stable_model_id": stable.model_id,
                "baseline_model_id": baseline.model_id,
                "metadata": stable.metadata,
                "failed": failed,
                "reasons": reasons,
                "log_loss_delta": round(log_loss_delta, 6),
                "brier_delta": round(brier_delta, 6),
                "max_calibration_error": round(max_calibration_error, 6),
            }
        )

    if consecutive_failures >= gate.required_consecutive_failures:
        decision = "rollback"
        reasons = [
            f"{consecutive_failures} consecutive windows failed against {baseline_reports[-1].model_id}"
        ]
    elif any(result["failed"] for result in window_results):
        decision = "watch"
        reasons = ["at least one evaluation window failed rollback checks"]
    else:
        decision = "keep"
        reasons = ["stable profile passed rollback checks"]

    return RollbackDecision(
        gate_id=gate.gate_id,
        stable_model_id=stable_reports[-1].model_id,
        baseline_model_id=baseline_reports[-1].model_id,
        decision=decision,
        reasons=reasons,
        metrics={
            "windows": len(window_results),
            "consecutive_failures": consecutive_failures,
            "window_results": window_results,
        },
    )


def _neutral(request: PredictionRequest, model_id: str) -> PredictionDistribution:
    outcomes = request.outcomes or ["yes", "no"]
    base = 1 / len(outcomes)
    return PredictionDistribution(outcomes={outcome: base for outcome in outcomes}, model_id=model_id)
