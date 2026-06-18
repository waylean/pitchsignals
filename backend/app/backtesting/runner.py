from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

from app.backtesting.records import FootballBacktestDataset, FootballMatchRecord
from app.models.governance import (
    EnsembleProfile,
    EvaluationCase,
    EvaluationReport,
    MultiWindowPromotionGate,
    ModelLeaderboard,
    PromotionDecision,
    PredictionDistribution,
    PromotionGate,
    build_evaluation_report,
    build_model_leaderboard,
    evaluate_multi_window_promotion_gate,
    evaluate_promotion_gate,
)


OUTCOMES = ["home_win", "draw", "away_win"]
CANDIDATE_MODEL_ID = "football_data_market_league_calibrated"
GLOBAL_CALIBRATED_MODEL_ID = "football_data_market_calibrated"
RAW_MARKET_MODEL_ID = "football_data_market_only"
PREVIOUS_STABLE_MODEL_ID = "previous_stable_profile"
FOOTBALL_MIN_CALIBRATION_BIN_CASES = 20


@dataclass(frozen=True)
class RollingWindowSpec:
    window_id: str
    train_seasons: list[str]
    validation_season: str
    test_season: str


@dataclass(frozen=True)
class RollingWindowResult:
    window: RollingWindowSpec
    manifest: dict[str, Any]
    split: dict[str, Any]
    market_temperature_alpha: float
    market_temperature_by_league: dict[str, float]
    train_reports: list[EvaluationReport]
    validation_reports: list[EvaluationReport]
    test_reports: list[EvaluationReport]
    test_leaderboard: ModelLeaderboard
    test_slice_reports: dict[str, dict[str, EvaluationReport]]
    decisions: dict[str, PromotionDecision]
    leakage_issue_count: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest,
            "window": asdict(self.window),
            "split": self.split,
            "market_temperature_alpha": self.market_temperature_alpha,
            "market_temperature_by_league": self.market_temperature_by_league,
            "train_reports": [asdict(report) for report in self.train_reports],
            "validation_reports": [asdict(report) for report in self.validation_reports],
            "test_reports": [asdict(report) for report in self.test_reports],
            "test_leaderboard": asdict(self.test_leaderboard),
            "test_slice_reports": {
                slice_id: {model_id: asdict(report) for model_id, report in reports.items()}
                for slice_id, reports in self.test_slice_reports.items()
            },
            "promotion_decision_vs_uniform": asdict(self.decisions["uniform_baseline"]),
            "promotion_decision_vs_home_prior": asdict(self.decisions["home_prior_baseline"]),
            "promotion_decision_vs_raw_market": asdict(self.decisions["football_data_market_only"]),
            **(
                {
                    "promotion_decision_vs_previous_stable": asdict(
                        self.decisions["previous_stable_profile"]
                    )
                }
                if "previous_stable_profile" in self.decisions
                else {}
            ),
            "leakage_issue_count": self.leakage_issue_count,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class MultiWindowRollingArtifact:
    artifact_id: str
    dataset_manifest: dict[str, Any]
    window_mode: str
    window_plan: list[RollingWindowSpec]
    windows: list[RollingWindowResult]
    summary: dict[str, Any]
    promotion_decision: PromotionDecision | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "manifest": self.dataset_manifest,
            "window_mode": self.window_mode,
            "window_plan": [asdict(window) for window in self.window_plan],
            "windows": [window.to_dict() for window in self.windows],
            "summary": self.summary,
            "candidate_model_id": self.metadata.get("candidate_model_id"),
            "rolling_promotion_decision": (
                asdict(self.promotion_decision) if self.promotion_decision else None
            ),
            "metadata": self.metadata,
        }


def football_data_market_backtest(dataset: FootballBacktestDataset) -> dict[str, Any]:
    uniform_cases = [
        _case(record, _uniform_distribution("uniform_baseline"), "uniform_baseline")
        for record in dataset.records
    ]
    market_cases = [
        _case(record, _market_distribution(record, "football_data_market_only"), "football_data_market_only")
        for record in dataset.records
    ]
    home_prior_cases = [
        _case(record, _home_prior_distribution("home_prior_baseline"), "home_prior_baseline")
        for record in dataset.records
    ]

    reports = [
        build_evaluation_report(
            "uniform_baseline",
            uniform_cases,
            {"dataset_id": dataset.manifest.dataset_id, "profile_type": "naive"},
        ),
        build_evaluation_report(
            "home_prior_baseline",
            home_prior_cases,
            {"dataset_id": dataset.manifest.dataset_id, "profile_type": "simple_prior"},
        ),
        build_evaluation_report(
            "football_data_market_only",
            market_cases,
            {"dataset_id": dataset.manifest.dataset_id, "profile_type": "market_only"},
        ),
    ]
    leaderboard = build_model_leaderboard(
        f"{dataset.manifest.dataset_id}_leaderboard",
        reports,
        metadata={
            "dataset_id": dataset.manifest.dataset_id,
            "competition": dataset.manifest.competition,
            "season": dataset.manifest.season,
            "market_available": True,
        },
    )
    promotion = evaluate_promotion_gate(
        reports[2],
        reports[0],
        PromotionGate(
            gate_id="football_data_market_minimum_gate",
            min_cases=250,
            max_log_loss_delta_vs_baseline=-0.01,
            max_brier_delta_vs_baseline=-0.005,
            max_calibration_error=0.18,
            min_calibration_bin_cases=FOOTBALL_MIN_CALIBRATION_BIN_CASES,
        ),
    )
    return {
        "manifest": asdict(dataset.manifest),
        "reports": [asdict(report) for report in reports],
        "leaderboard": asdict(leaderboard),
        "promotion_decision": asdict(promotion),
        "sample_matches": [_record_summary(record) for record in dataset.records[:10]],
    }


def build_rolling_windows(
    seasons: list[str],
    train_window_size: int,
    mode: str = "sliding",
) -> list[RollingWindowSpec]:
    if train_window_size < 1:
        raise ValueError("train_window_size must be >= 1")
    if len(seasons) < train_window_size + 2:
        raise ValueError("rolling windows require at least train_window_size + 2 seasons")
    if mode not in {"sliding", "expanding"}:
        raise ValueError("mode must be 'sliding' or 'expanding'")

    windows: list[RollingWindowSpec] = []
    for validation_index in range(train_window_size, len(seasons) - 1):
        if mode == "expanding":
            train_seasons = seasons[:validation_index]
        else:
            train_seasons = seasons[validation_index - train_window_size : validation_index]
        validation_season = seasons[validation_index]
        test_season = seasons[validation_index + 1]
        windows.append(
            RollingWindowSpec(
                window_id=(
                    f"train_{'-'.join(train_seasons)}__"
                    f"val_{validation_season}__test_{test_season}"
                ),
                train_seasons=train_seasons,
                validation_season=validation_season,
                test_season=test_season,
            )
        )
    return windows


def football_data_multi_window_rolling_backtest(
    dataset: FootballBacktestDataset,
    seasons: list[str],
    train_window_size: int = 2,
    mode: str = "sliding",
    previous_stable_profile: EnsembleProfile | None = None,
) -> dict[str, Any]:
    return football_data_multi_window_rolling_artifact(
        dataset,
        seasons=seasons,
        train_window_size=train_window_size,
        mode=mode,
        previous_stable_profile=previous_stable_profile,
    ).to_dict()


def football_data_multi_window_rolling_artifact(
    dataset: FootballBacktestDataset,
    seasons: list[str],
    train_window_size: int = 2,
    mode: str = "sliding",
    previous_stable_profile: EnsembleProfile | None = None,
) -> MultiWindowRollingArtifact:
    windows = build_rolling_windows(seasons, train_window_size, mode=mode)
    results = [
        _football_data_rolling_window_result(
            dataset,
            window=window,
            previous_stable_profile=previous_stable_profile,
        )
        for window in windows
    ]
    summary = summarize_multi_window_result({"windows": [result.to_dict() for result in results]})
    artifact = MultiWindowRollingArtifact(
        artifact_id=f"{dataset.manifest.dataset_id}_multi_window_{mode}",
        dataset_manifest=asdict(dataset.manifest),
        window_mode=mode,
        window_plan=windows,
        windows=results,
        summary=summary,
        promotion_decision=None,
        metadata={
            "horizon_policy": dataset.manifest.horizon_policy,
            "snapshot_policy": dataset.manifest.snapshot_policy,
            "candidate_model_id": CANDIDATE_MODEL_ID,
            "previous_stable_profile": (
                {
                    "profile_id": previous_stable_profile.profile_id,
                    "version": previous_stable_profile.version,
                    "maturity": previous_stable_profile.maturity,
                    "horizon_profile": previous_stable_profile.horizon_profile,
                }
                if previous_stable_profile
                else None
            ),
        },
    )
    promotion_decision = evaluate_multi_window_promotion_gate(
        artifact,
        MultiWindowPromotionGate(
            gate_id="football_data_multi_window_market_proxy_gate",
            min_windows=3,
            min_total_test_cases=5000,
            require_zero_leakage=True,
            latest_window_must_pass=True,
        ),
    )
    return replace(artifact, promotion_decision=promotion_decision)


def football_data_sliced_market_backtest(dataset: FootballBacktestDataset) -> dict[str, Any]:
    overall = football_data_market_backtest(dataset)
    slices: list[dict[str, Any]] = []
    for competition in sorted({record.competition for record in dataset.records}):
        records = [record for record in dataset.records if record.competition == competition]
        slices.append(
            football_data_market_backtest(
                FootballBacktestDataset(
                    manifest=_slice_manifest(dataset, f"competition:{competition}", records),
                    records=records,
                )
            )
        )
    for season in sorted({record.season for record in dataset.records}):
        records = [record for record in dataset.records if record.season == season]
        slices.append(
            football_data_market_backtest(
                FootballBacktestDataset(
                    manifest=_slice_manifest(dataset, f"season:{season}", records),
                    records=records,
                )
            )
        )
    overall["slices"] = slices
    return overall


def football_data_rolling_market_backtest(
    dataset: FootballBacktestDataset,
    train_seasons: list[str],
    validation_season: str,
    test_season: str,
    previous_stable_profile: EnsembleProfile | None = None,
) -> dict[str, Any]:
    return _football_data_rolling_window_result(
        dataset,
        RollingWindowSpec(
            window_id=(
                f"train_{'-'.join(train_seasons)}__"
                f"val_{validation_season}__test_{test_season}"
            ),
            train_seasons=train_seasons,
            validation_season=validation_season,
            test_season=test_season,
        ),
        previous_stable_profile=previous_stable_profile,
    ).to_dict()


def _football_data_rolling_window_result(
    dataset: FootballBacktestDataset,
    window: RollingWindowSpec,
    previous_stable_profile: EnsembleProfile | None = None,
) -> RollingWindowResult:
    split_id = window.window_id
    train_records = [
        record for record in dataset.records if record.season in set(window.train_seasons)
    ]
    validation_records = [
        record for record in dataset.records if record.season == window.validation_season
    ]
    test_records = [
        record for record in dataset.records if record.season == window.test_season
    ]
    alpha = train_market_temperature(train_records)
    league_alphas = train_league_market_temperatures(train_records, fallback_alpha=alpha)

    train_reports = _rolling_reports(train_records, "train", alpha, league_alphas, previous_stable_profile)
    validation_reports = _rolling_reports(
        validation_records,
        "validation",
        alpha,
        league_alphas,
        previous_stable_profile,
    )
    test_reports = _rolling_reports(test_records, "test", alpha, league_alphas, previous_stable_profile)
    test_slice_reports = _rolling_slice_reports(test_records, alpha, league_alphas, previous_stable_profile)
    test_leaderboard = build_model_leaderboard(
        f"{dataset.manifest.dataset_id}_{split_id}_leaderboard",
        test_reports,
        metadata={
            "dataset_id": dataset.manifest.dataset_id,
            "split_id": split_id,
            "train_seasons": window.train_seasons,
            "validation_season": window.validation_season,
            "test_season": window.test_season,
            "market_available": True,
        },
    )
    by_model = {report.model_id: report for report in test_reports}
    promotion = evaluate_promotion_gate(
        by_model[CANDIDATE_MODEL_ID],
        by_model["uniform_baseline"],
        PromotionGate(
            gate_id="football_data_rolling_market_gate",
            min_cases=250,
            max_log_loss_delta_vs_baseline=-0.01,
            max_brier_delta_vs_baseline=-0.005,
            max_calibration_error=0.18,
            min_calibration_bin_cases=FOOTBALL_MIN_CALIBRATION_BIN_CASES,
        ),
    )
    home_prior_comparison = evaluate_promotion_gate(
        by_model[CANDIDATE_MODEL_ID],
        by_model["home_prior_baseline"],
        PromotionGate(
            gate_id="football_data_market_calibration_vs_home_prior_gate",
            min_cases=250,
            max_log_loss_delta_vs_baseline=-0.01,
            max_brier_delta_vs_baseline=-0.005,
            max_calibration_error=0.18,
            min_calibration_bin_cases=FOOTBALL_MIN_CALIBRATION_BIN_CASES,
        ),
    )
    raw_market_comparison = evaluate_promotion_gate(
        by_model[CANDIDATE_MODEL_ID],
        by_model[RAW_MARKET_MODEL_ID],
        PromotionGate(
            gate_id="football_data_market_calibration_vs_raw_gate",
            min_cases=250,
            max_log_loss_delta_vs_baseline=0.002,
            max_brier_delta_vs_baseline=0.002,
            max_calibration_error=0.18,
            min_calibration_bin_cases=FOOTBALL_MIN_CALIBRATION_BIN_CASES,
            required_slices=sorted(test_slice_reports),
        ),
        slice_reports={
            slice_id: (
                reports[CANDIDATE_MODEL_ID],
                reports[RAW_MARKET_MODEL_ID],
            )
            for slice_id, reports in test_slice_reports.items()
        },
    )
    decisions = {
        "uniform_baseline": promotion,
        "home_prior_baseline": home_prior_comparison,
        RAW_MARKET_MODEL_ID: raw_market_comparison,
    }
    if previous_stable_profile and PREVIOUS_STABLE_MODEL_ID in by_model:
        previous_slice_reports = {
            slice_id: (
                reports[CANDIDATE_MODEL_ID],
                reports[PREVIOUS_STABLE_MODEL_ID],
            )
            for slice_id, reports in test_slice_reports.items()
            if PREVIOUS_STABLE_MODEL_ID in reports
        }
        previous_stable_comparison = evaluate_promotion_gate(
            by_model[CANDIDATE_MODEL_ID],
            by_model[PREVIOUS_STABLE_MODEL_ID],
            PromotionGate(
                gate_id="football_data_market_calibration_vs_previous_stable_gate",
                min_cases=250,
                max_log_loss_delta_vs_baseline=0.0,
                max_brier_delta_vs_baseline=0.0,
                max_calibration_error=0.18,
                min_calibration_bin_cases=FOOTBALL_MIN_CALIBRATION_BIN_CASES,
                required_slices=sorted(previous_slice_reports),
            ),
            slice_reports=previous_slice_reports,
        )
        decisions[PREVIOUS_STABLE_MODEL_ID] = previous_stable_comparison
    leakage_issue_count = sum(len(record.leakage_issues()) for record in test_records)
    return RollingWindowResult(
        window=window,
        manifest=asdict(dataset.manifest),
        split={
            "split_id": split_id,
            "train_seasons": window.train_seasons,
            "validation_season": window.validation_season,
            "test_season": window.test_season,
            "train_cases": len(train_records),
            "validation_cases": len(validation_records),
            "test_cases": len(test_records),
        },
        market_temperature_alpha=alpha,
        market_temperature_by_league=league_alphas,
        train_reports=train_reports,
        validation_reports=validation_reports,
        test_reports=test_reports,
        test_leaderboard=test_leaderboard,
        test_slice_reports=test_slice_reports,
        decisions=decisions,
        leakage_issue_count=leakage_issue_count,
        metadata={
            "horizon_policy": dataset.manifest.horizon_policy,
            "snapshot_policy": dataset.manifest.snapshot_policy,
            "previous_stable_profile": (
                {
                    "profile_id": previous_stable_profile.profile_id,
                    "version": previous_stable_profile.version,
                    "maturity": previous_stable_profile.maturity,
                    "horizon_profile": previous_stable_profile.horizon_profile,
                }
                if previous_stable_profile
                else None
            ),
        },
    )


def train_market_temperature(records: list[FootballMatchRecord]) -> float:
    candidates = [round(0.70 + index * 0.02, 2) for index in range(41)]
    best_alpha = 1.0
    best_loss = float("inf")
    for alpha in candidates:
        report = _report_for_distribution(
            records,
            f"market_temperature_{alpha}",
            lambda record, model_id: _calibrated_market_distribution(record, model_id, alpha),
            {"split": "train", "alpha": alpha},
        )
        if report.log_loss < best_loss:
            best_loss = report.log_loss
            best_alpha = alpha
    return best_alpha


def train_league_market_temperatures(
    records: list[FootballMatchRecord],
    fallback_alpha: float,
    min_cases: int = 250,
) -> dict[str, float]:
    league_alphas: dict[str, float] = {}
    for competition in sorted({record.competition for record in records}):
        league_records = [record for record in records if record.competition == competition]
        if len(league_records) < min_cases:
            league_alphas[competition] = fallback_alpha
            continue
        league_alphas[competition] = train_market_temperature(league_records)
    return league_alphas


def summarize_multi_window_result(result: dict[str, Any]) -> dict[str, Any]:
    windows = result["windows"]
    raw_market_decisions = [
        window["promotion_decision_vs_raw_market"]["decision"] for window in windows
    ]
    previous_stable_decisions = [
        window["promotion_decision_vs_previous_stable"]["decision"]
        for window in windows
        if "promotion_decision_vs_previous_stable" in window
    ]
    uniform_decisions = [
        window["promotion_decision_vs_uniform"]["decision"] for window in windows
    ]
    calibrated_reports = [
        _report_by_model(window["test_reports"], CANDIDATE_MODEL_ID)
        for window in windows
    ]
    raw_reports = [
        _report_by_model(window["test_reports"], RAW_MARKET_MODEL_ID)
        for window in windows
    ]
    leakage_issues = sum(
        int(window["manifest"].get("coverage", {}).get("leakage_issues") or 0)
        for window in windows
    )
    return {
        "window_count": len(windows),
        "test_cases": sum(window["split"]["test_cases"] for window in windows),
        "uniform_promotions": uniform_decisions.count("promote"),
        "raw_market_promotions": raw_market_decisions.count("promote"),
        "raw_market_holds": raw_market_decisions.count("hold"),
        "previous_stable_promotions": previous_stable_decisions.count("promote"),
        "previous_stable_holds": previous_stable_decisions.count("hold"),
        "weighted_calibrated_log_loss": _weighted_average(calibrated_reports, "log_loss"),
        "weighted_raw_market_log_loss": _weighted_average(raw_reports, "log_loss"),
        "weighted_calibrated_brier": _weighted_average(calibrated_reports, "brier_score"),
        "weighted_raw_market_brier": _weighted_average(raw_reports, "brier_score"),
        "leakage_issues": leakage_issues,
    }


def rolling_promotion_decision(result: dict[str, Any]) -> dict[str, Any]:
    decision = evaluate_multi_window_promotion_gate(
        result,
        MultiWindowPromotionGate(
            gate_id="football_data_multi_window_market_proxy_gate",
            min_windows=3,
            min_total_test_cases=5000,
            require_zero_leakage=True,
            latest_window_must_pass=True,
        ),
    )
    return asdict(decision)


def _report_by_model(reports: list[dict[str, Any]], model_id: str) -> dict[str, Any]:
    for report in reports:
        if report["model_id"] == model_id:
            return report
    raise KeyError(model_id)


def _weighted_average(reports: list[dict[str, Any]], metric: str) -> float:
    total_cases = sum(report["case_count"] for report in reports)
    if total_cases <= 0:
        return 0.0
    return round(
        sum(report[metric] * report["case_count"] for report in reports) / total_cases,
        6,
    )


def _case(
    record: FootballMatchRecord,
    distribution: PredictionDistribution,
    model_id: str,
) -> EvaluationCase:
    return EvaluationCase(
        actual=record.actual_result,
        distribution=distribution,
        horizon_profile="closing_or_matchday_snapshot",
        metadata={
            "match_id": record.match_id,
            "model_id": model_id,
            "competition": record.competition,
            "season": record.season,
            "market_available": record.has_1x2_odds,
            "prediction_deadline": (
                record.prediction_deadline.isoformat() if record.prediction_deadline else None
            ),
            "odds_as_of": record.odds_as_of.isoformat() if record.odds_as_of else None,
            "odds_available_at": (
                record.odds_available_at.isoformat() if record.odds_available_at else None
            ),
            "feature_available_at": (
                record.feature_available_at.isoformat() if record.feature_available_at else None
            ),
            "source_snapshot_id": record.source_snapshot_id,
            "odds_snapshot_type": record.odds_snapshot_type,
            "leakage_issues": record.leakage_issues(),
        },
    )


def _rolling_reports(
    records: list[FootballMatchRecord],
    split: str,
    alpha: float,
    league_alphas: dict[str, float] | None = None,
    previous_stable_profile: EnsembleProfile | None = None,
) -> list[EvaluationReport]:
    reports = [
        _report_for_distribution(
            records,
            "uniform_baseline",
            lambda _record, model_id: _uniform_distribution(model_id),
            {"split": split},
        ),
        _report_for_distribution(
            records,
            "home_prior_baseline",
            lambda _record, model_id: _home_prior_distribution(model_id),
            {"split": split},
        ),
        _report_for_distribution(
            records,
            RAW_MARKET_MODEL_ID,
            _market_distribution,
            {"split": split},
        ),
        _report_for_distribution(
            records,
            GLOBAL_CALIBRATED_MODEL_ID,
            lambda record, model_id: _calibrated_market_distribution(record, model_id, alpha),
            {"split": split, "alpha": alpha},
        ),
    ]
    if league_alphas is not None:
        reports.append(
            _report_for_distribution(
                records,
                CANDIDATE_MODEL_ID,
                lambda record, model_id: _league_calibrated_market_distribution(
                    record,
                    model_id,
                    league_alphas,
                    alpha,
                ),
                {
                    "split": split,
                    "fallback_alpha": alpha,
                    "league_alphas": league_alphas,
                },
            )
        )
    if previous_stable_profile is not None:
        reports.append(
            _report_for_distribution(
                records,
                PREVIOUS_STABLE_MODEL_ID,
                lambda record, model_id: _previous_stable_profile_distribution(
                    record,
                    model_id,
                    previous_stable_profile,
                    alpha,
                    league_alphas,
                ),
                {
                    "split": split,
                    "profile_id": previous_stable_profile.profile_id,
                    "profile_version": previous_stable_profile.version,
                    "profile_maturity": previous_stable_profile.maturity,
                    "horizon_profile": previous_stable_profile.horizon_profile,
                },
            )
        )
    return reports


def _rolling_slice_reports(
    records: list[FootballMatchRecord],
    alpha: float,
    league_alphas: dict[str, float] | None = None,
    previous_stable_profile: EnsembleProfile | None = None,
    min_cases: int = 250,
) -> dict[str, dict[str, EvaluationReport]]:
    results: dict[str, dict[str, EvaluationReport]] = {}
    slice_groups: dict[str, list[FootballMatchRecord]] = {}
    for record in records:
        _append_slice(slice_groups, f"league:{record.competition}", record)
        _append_slice(slice_groups, f"season:{record.season}", record)
        _append_slice(slice_groups, f"league_season:{record.competition}:{record.season}", record)
        _append_slice(slice_groups, f"horizon:{record.odds_snapshot_type}", record)
        _append_slice(slice_groups, f"market_available:{str(record.has_1x2_odds).lower()}", record)
    for slice_id, slice_records in sorted(slice_groups.items()):
        if len(slice_records) < min_cases:
            continue
        reports = _rolling_reports(
            slice_records,
            f"test:{slice_id}",
            alpha,
            league_alphas,
            previous_stable_profile,
        )
        results[slice_id] = {report.model_id: report for report in reports}
    return results


def _append_slice(
    slice_groups: dict[str, list[FootballMatchRecord]],
    slice_id: str,
    record: FootballMatchRecord,
) -> None:
    slice_groups.setdefault(slice_id, []).append(record)


def _report_for_distribution(
    records: list[FootballMatchRecord],
    model_id: str,
    distribution_factory,
    metadata: dict[str, Any],
) -> EvaluationReport:
    cases = [_case(record, distribution_factory(record, model_id), model_id) for record in records]
    return build_evaluation_report(model_id, cases, metadata)


def _uniform_distribution(model_id: str) -> PredictionDistribution:
    return PredictionDistribution(
        model_id=model_id,
        outcomes={outcome: 1 / 3 for outcome in OUTCOMES},
    )


def _home_prior_distribution(model_id: str) -> PredictionDistribution:
    return PredictionDistribution(
        model_id=model_id,
        outcomes={"home_win": 0.44, "draw": 0.27, "away_win": 0.29},
    )


def _market_distribution(record: FootballMatchRecord, model_id: str) -> PredictionDistribution:
    raw = {
        "home_win": 1 / record.odds["home"],
        "draw": 1 / record.odds["draw"],
        "away_win": 1 / record.odds["away"],
    }
    total = sum(raw.values()) or 1
    return PredictionDistribution(
        model_id=model_id,
        outcomes={key: value / total for key, value in raw.items()},
        rationale=f"Football-Data overround-adjusted {record.odds.get('source', 'unknown')} 1X2 odds.",
    )


def _calibrated_market_distribution(
    record: FootballMatchRecord,
    model_id: str,
    alpha: float,
) -> PredictionDistribution:
    raw_market = _market_distribution(record, model_id).outcomes
    adjusted = {key: max(value, 1e-9) ** alpha for key, value in raw_market.items()}
    total = sum(adjusted.values()) or 1
    return PredictionDistribution(
        model_id=model_id,
        outcomes={key: value / total for key, value in adjusted.items()},
        rationale=f"Football-Data market odds calibrated with alpha={alpha:.2f}.",
    )


def _league_calibrated_market_distribution(
    record: FootballMatchRecord,
    model_id: str,
    league_alphas: dict[str, float],
    fallback_alpha: float,
) -> PredictionDistribution:
    alpha = league_alphas.get(record.competition, fallback_alpha)
    distribution = _calibrated_market_distribution(record, model_id, alpha)
    return PredictionDistribution(
        model_id=model_id,
        outcomes=distribution.outcomes,
        rationale=(
            f"Football-Data market odds calibrated with league alpha={alpha:.2f} "
            f"for {record.competition}."
        ),
    )


def _previous_stable_profile_distribution(
    record: FootballMatchRecord,
    model_id: str,
    profile: EnsembleProfile,
    alpha: float,
    league_alphas: dict[str, float] | None,
) -> PredictionDistribution:
    by_model = {
        "uniform_baseline": _uniform_distribution("uniform_baseline"),
        "home_prior_baseline": _home_prior_distribution("home_prior_baseline"),
        RAW_MARKET_MODEL_ID: _market_distribution(record, RAW_MARKET_MODEL_ID),
        GLOBAL_CALIBRATED_MODEL_ID: _calibrated_market_distribution(
            record,
            GLOBAL_CALIBRATED_MODEL_ID,
            alpha,
        ),
    }
    if league_alphas is not None:
        by_model[CANDIDATE_MODEL_ID] = _league_calibrated_market_distribution(
            record,
            CANDIDATE_MODEL_ID,
            league_alphas,
            alpha,
        )
    weights = profile.normalized_weights(set(by_model))
    if not weights:
        return PredictionDistribution(
            model_id=model_id,
            outcomes=_uniform_distribution(model_id).outcomes,
            rationale=(
                f"Previous stable profile {profile.profile_id}@{profile.version} had no "
                "usable model weights in this runner; fell back to uniform."
            ),
        )
    outcomes = {
        outcome: sum(
            by_model[weighted_model_id].normalized().outcomes.get(outcome, 0.0) * weight
            for weighted_model_id, weight in weights.items()
        )
        for outcome in OUTCOMES
    }
    total = sum(outcomes.values()) or 1.0
    return PredictionDistribution(
        model_id=model_id,
        outcomes={outcome: value / total for outcome, value in outcomes.items()},
        rationale=(
            f"Previous stable profile {profile.profile_id}@{profile.version} "
            f"blended from {', '.join(sorted(weights))}."
        ),
    )


def _record_summary(record: FootballMatchRecord) -> dict[str, Any]:
    return {
        "match_id": record.match_id,
        "date": record.event_time.date().isoformat(),
        "match": f"{record.home_team} vs {record.away_team}",
        "score": f"{record.home_goals}-{record.away_goals}",
        "actual": record.actual_result,
        "odds": record.odds,
    }


def _slice_manifest(
    dataset: FootballBacktestDataset,
    slice_id: str,
    records: list[FootballMatchRecord],
):
    from dataclasses import replace

    return replace(
        dataset.manifest,
        dataset_id=f"{dataset.manifest.dataset_id}_{slice_id.replace(':', '_')}",
        competition=slice_id,
        season=None,
        case_count=len(records),
        coverage={
            **dataset.manifest.coverage,
            "slice_id": slice_id,
            "records_with_1x2_odds": sum(1 for record in records if record.has_1x2_odds),
        },
        gaps=[],
    )
