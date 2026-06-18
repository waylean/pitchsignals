from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.data_sources.football_dataset_registry import (
    football_1_0_dataset_matrix,
    required_feature_coverage,
)
from app.backtesting.records import DatasetManifest, FootballBacktestDataset, FootballMatchRecord
from app.backtesting.runner import (
    MultiWindowRollingArtifact,
    build_rolling_windows,
    football_data_multi_window_rolling_artifact,
    football_data_rolling_market_backtest,
)
from app.models.governance import (
    EnsembleProfile,
    EvaluationReport,
    MultiWindowPromotionGate,
    PromotionGate,
    RollbackGate,
    evaluate_multi_window_promotion_gate,
    evaluate_promotion_gate,
    evaluate_rollback_gate,
    get_default_ensemble_profile,
    load_ensemble_profiles,
    load_profile_lifecycle_artifacts,
)


def main() -> None:
    matrix = football_1_0_dataset_matrix()
    coverage = required_feature_coverage()
    assert len(matrix) >= 6
    assert "market_odds" in coverage
    assert "xg_form" in coverage
    assert "lineup_availability" in coverage

    windows = build_rolling_windows(["1920", "2021", "2122", "2223", "2324"], 2)
    assert [window.window_id for window in windows] == [
        "train_1920-2021__val_2122__test_2223",
        "train_2021-2122__val_2223__test_2324",
    ]
    empty_artifact = football_data_multi_window_rolling_artifact(
        FootballBacktestDataset(
            manifest=DatasetManifest(
                dataset_id="empty_smoke",
                source_url="local",
                license_note="smoke",
                case_count=0,
            ),
            records=[],
        ),
        seasons=["1920", "2021", "2122"],
        train_window_size=1,
    )
    assert isinstance(empty_artifact, MultiWindowRollingArtifact)
    empty_payload = empty_artifact.to_dict()
    assert "windows" in empty_payload
    assert "rolling_promotion_decision" in empty_payload
    typed_decision = evaluate_multi_window_promotion_gate(
        empty_artifact,
        MultiWindowPromotionGate(
            gate_id="smoke_typed_artifact",
            min_windows=1,
            min_total_test_cases=1,
        ),
    )
    assert typed_decision.decision == "hold"

    profiles = load_ensemble_profiles()
    assert "football" in profiles
    assert any(profile.profile_id == "football_market_proxy_candidate" for profile in profiles["football"])
    candidate_profile = next(
        profile for profile in profiles["football"] if profile.profile_id == "football_market_proxy_candidate"
    )
    assert candidate_profile.version == "0.7.1"
    assert "football_data_market_league_calibrated" in candidate_profile.model_weights
    default_profile = get_default_ensemble_profile("football")
    assert default_profile is not None
    assert default_profile.profile_id == "football_research_alpha"
    assert default_profile.test_report_id == "football_data_rolling_backtest"
    assert get_default_ensemble_profile("football", include_research_profiles=False) is None
    lifecycle_artifacts = load_profile_lifecycle_artifacts()
    assert "football" in lifecycle_artifacts
    assert any(
        artifact.profile_id == "football_market_proxy_candidate"
        and artifact.profile_version == "0.7.1"
        and artifact.status == "market_proxy_promoted_not_stable"
        for artifact in lifecycle_artifacts["football"]
    )

    baseline = EvaluationReport(
        model_id="baseline",
        case_count=48,
        log_loss=1.094302,
        brier_score=0.663696,
        ranked_probability_score=0.241562,
        accuracy=0.520833,
        calibration_bins=[{"count": 48, "calibration_error": 0.17}],
    )
    candidate = EvaluationReport(
        model_id="candidate",
        case_count=48,
        log_loss=1.088358,
        brier_score=0.658302,
        ranked_probability_score=0.23887,
        accuracy=0.520833,
        calibration_bins=[{"count": 48, "calibration_error": 0.17}],
    )
    decision = evaluate_promotion_gate(
        candidate,
        baseline,
        PromotionGate(gate_id="smoke", min_cases=250),
    )
    assert decision.decision == "hold"
    assert any("case_count" in reason for reason in decision.reasons)

    strong_candidate = EvaluationReport(
        model_id="market_candidate",
        case_count=5403,
        log_loss=0.970453,
        brier_score=0.57705,
        ranked_probability_score=0.194908,
        accuracy=0.53933,
        calibration_bins=[{"count": 5403, "calibration_error": 0.0286}],
    )
    weak_baseline = EvaluationReport(
        model_id="uniform",
        case_count=5403,
        log_loss=1.098712,
        brier_score=0.666667,
        ranked_probability_score=0.235276,
        accuracy=0.438645,
        calibration_bins=[{"count": 5403, "calibration_error": 0.17}],
    )
    promote = evaluate_promotion_gate(
        strong_candidate,
        weak_baseline,
        PromotionGate(gate_id="smoke_promote", min_cases=250, max_calibration_error=0.18),
    )
    assert promote.decision == "promote"

    slice_hold = evaluate_promotion_gate(
        strong_candidate,
        weak_baseline,
        PromotionGate(
            gate_id="smoke_required_slice",
            min_cases=250,
            max_calibration_error=0.18,
            required_slices=["league:E0"],
        ),
    )
    assert slice_hold.decision == "hold"
    assert "required slice league:E0 is missing" in slice_hold.reasons

    slice_promote = evaluate_promotion_gate(
        strong_candidate,
        weak_baseline,
        PromotionGate(
            gate_id="smoke_required_slice_pass",
            min_cases=250,
            max_calibration_error=0.18,
            required_slices=["league:E0"],
        ),
        slice_reports={"league:E0": (strong_candidate, weak_baseline)},
    )
    assert slice_promote.decision == "promote"

    sparse_bin_candidate = EvaluationReport(
        model_id="sparse_bin_candidate",
        case_count=500,
        log_loss=0.90,
        brier_score=0.52,
        ranked_probability_score=0.18,
        accuracy=0.55,
        calibration_bins=[
            {"lower": 0.4, "upper": 0.6, "count": 499, "calibration_error": 0.03},
            {"lower": 0.8, "upper": 1.0, "count": 1, "calibration_error": 0.82},
        ],
    )
    sparse_bin_promote = evaluate_promotion_gate(
        sparse_bin_candidate,
        weak_baseline,
        PromotionGate(
            gate_id="smoke_sparse_bin_supported_calibration",
            min_cases=250,
            max_calibration_error=0.18,
            min_calibration_bin_cases=20,
        ),
    )
    assert sparse_bin_promote.decision == "promote"
    assert (
        sparse_bin_promote.metrics["max_calibration_error_all_bins"]
        > sparse_bin_promote.metrics["max_supported_calibration_error"]
    )
    assert sparse_bin_promote.metrics["ignored_sparse_calibration_bins"]

    supported_bin_candidate = EvaluationReport(
        model_id="supported_bin_candidate",
        case_count=500,
        log_loss=0.90,
        brier_score=0.52,
        ranked_probability_score=0.18,
        accuracy=0.55,
        calibration_bins=[
            {"lower": 0.4, "upper": 0.6, "count": 460, "calibration_error": 0.03},
            {"lower": 0.8, "upper": 1.0, "count": 40, "calibration_error": 0.32},
        ],
    )
    supported_bin_hold = evaluate_promotion_gate(
        supported_bin_candidate,
        weak_baseline,
        PromotionGate(
            gate_id="smoke_supported_bin_hold",
            min_cases=250,
            max_calibration_error=0.18,
            min_calibration_bin_cases=20,
        ),
    )
    assert supported_bin_hold.decision == "hold"
    assert any("max_calibration_error" in reason for reason in supported_bin_hold.reasons)

    multi_window_hold = evaluate_multi_window_promotion_gate(
        {
            "candidate_model_id": "football_data_market_calibrated",
            "summary": {
                "window_count": 2,
                "test_cases": 3578,
                "leakage_issues": 0,
            },
            "windows": [
                {
                    "split": {"split_id": "w1"},
                    "promotion_decision_vs_uniform": {"decision": "promote"},
                    "promotion_decision_vs_home_prior": {"decision": "promote"},
                    "promotion_decision_vs_raw_market": {
                        "decision": "promote",
                        "metrics": {
                            "slice_results": {
                                "league:premier_league": {"decision": "promote"},
                                "season:2223": {"decision": "promote"},
                                "league_season:premier_league:2223": {"decision": "promote"},
                                "horizon:market_snapshot_unknown_or_closing_proxy": {
                                    "decision": "promote"
                                },
                                "market_available:true": {"decision": "promote"},
                            }
                        },
                    },
                },
                {
                    "split": {"split_id": "w2"},
                    "promotion_decision_vs_uniform": {"decision": "promote"},
                    "promotion_decision_vs_home_prior": {"decision": "promote"},
                    "promotion_decision_vs_raw_market": {
                        "decision": "hold",
                        "metrics": {
                            "slice_results": {
                                "league:serie_a": {
                                    "decision": "hold",
                                    "reasons": ["calibration failed"],
                                },
                                "season:2324": {"decision": "promote"},
                                "league_season:serie_a:2324": {"decision": "hold"},
                                "horizon:market_snapshot_unknown_or_closing_proxy": {
                                    "decision": "promote"
                                },
                                "market_available:true": {"decision": "promote"},
                            }
                        },
                    },
                },
            ],
        },
        MultiWindowPromotionGate(
            gate_id="smoke_multi_window",
            min_windows=2,
            min_total_test_cases=1000,
        ),
    )
    assert multi_window_hold.decision == "hold"
    assert any("latest-window" in reason for reason in multi_window_hold.reasons)
    assert multi_window_hold.metrics["failed_required_slice_count"] == 1

    previous_stable_missing = evaluate_multi_window_promotion_gate(
        {
            "candidate_model_id": "football_horizon_candidate",
            "summary": {
                "window_count": 1,
                "test_cases": 500,
                "leakage_issues": 0,
            },
            "windows": [
                {
                    "split": {"split_id": "w1"},
                    "promotion_decision_vs_uniform": {"decision": "promote"},
                    "promotion_decision_vs_home_prior": {"decision": "promote"},
                    "promotion_decision_vs_raw_market": {
                        "decision": "promote",
                        "metrics": {
                            "slice_results": {
                                "league:premier_league": {"decision": "promote"},
                                "season:2324": {"decision": "promote"},
                                "league_season:premier_league:2324": {"decision": "promote"},
                                "horizon:T-24h": {"decision": "promote"},
                                "market_available:true": {"decision": "promote"},
                            }
                        },
                    },
                }
            ],
        },
        MultiWindowPromotionGate(
            gate_id="smoke_previous_stable_required_missing",
            min_windows=1,
            min_total_test_cases=100,
            required_baselines=[
                "uniform_baseline",
                "home_prior_baseline",
                "football_data_market_only",
                "previous_stable_profile",
            ],
        ),
    )
    assert previous_stable_missing.decision == "hold"
    assert any("previous_stable_profile" in reason for reason in previous_stable_missing.reasons)

    previous_stable_promote = evaluate_multi_window_promotion_gate(
        {
            "candidate_model_id": "football_horizon_candidate",
            "summary": {
                "window_count": 1,
                "test_cases": 500,
                "leakage_issues": 0,
            },
            "windows": [
                {
                    "split": {"split_id": "w1"},
                    "promotion_decision_vs_uniform": {"decision": "promote"},
                    "promotion_decision_vs_home_prior": {"decision": "promote"},
                    "promotion_decision_vs_raw_market": {
                        "decision": "promote",
                        "metrics": {
                            "slice_results": {
                                "league:premier_league": {"decision": "promote"},
                                "season:2324": {"decision": "promote"},
                                "league_season:premier_league:2324": {"decision": "promote"},
                                "horizon:T-24h": {"decision": "promote"},
                                "market_available:true": {"decision": "promote"},
                            }
                        },
                    },
                    "promotion_decision_vs_previous_stable": {"decision": "promote"},
                }
            ],
        },
        MultiWindowPromotionGate(
            gate_id="smoke_previous_stable_required_pass",
            min_windows=1,
            min_total_test_cases=100,
            required_baselines=[
                "uniform_baseline",
                "home_prior_baseline",
                "football_data_market_only",
                "previous_stable_profile",
            ],
        ),
    )
    assert previous_stable_promote.decision == "promote"
    assert previous_stable_promote.metrics["baseline_decisions"]["previous_stable_profile"] == [
        "promote"
    ]

    rolling_with_previous = football_data_rolling_market_backtest(
        _tiny_rolling_dataset(),
        train_seasons=["2122"],
        validation_season="2223",
        test_season="2324",
        previous_stable_profile=EnsembleProfile(
            profile_id="football_previous_stable_smoke",
            version="1.0.0",
            domain="football",
            maturity="stable",
            horizon_profile="T-24h",
            model_weights={"football_data_market_only": 1.0},
        ),
    )
    assert any(
        report["model_id"] == "previous_stable_profile"
        for report in rolling_with_previous["test_reports"]
    )
    assert "promotion_decision_vs_previous_stable" in rolling_with_previous
    assert (
        rolling_with_previous["promotion_decision_vs_previous_stable"]["baseline_model_id"]
        == "previous_stable_profile"
    )

    rollback = evaluate_rollback_gate(
        stable_reports=[
            EvaluationReport(
                model_id="stable_profile",
                case_count=300,
                log_loss=1.02,
                brier_score=0.61,
                ranked_probability_score=0.21,
                accuracy=0.50,
                calibration_bins=[{"count": 300, "calibration_error": 0.14}],
                metadata={"window": "w1"},
            ),
            EvaluationReport(
                model_id="stable_profile",
                case_count=300,
                log_loss=1.03,
                brier_score=0.62,
                ranked_probability_score=0.22,
                accuracy=0.49,
                calibration_bins=[{"count": 300, "calibration_error": 0.15}],
                metadata={"window": "w2"},
            ),
        ],
        baseline_reports=[
            EvaluationReport(
                model_id="market_only",
                case_count=300,
                log_loss=1.00,
                brier_score=0.60,
                ranked_probability_score=0.20,
                accuracy=0.51,
                calibration_bins=[{"count": 300, "calibration_error": 0.05}],
            ),
            EvaluationReport(
                model_id="market_only",
                case_count=300,
                log_loss=1.00,
                brier_score=0.60,
                ranked_probability_score=0.20,
                accuracy=0.51,
                calibration_bins=[{"count": 300, "calibration_error": 0.05}],
            ),
        ],
        gate=RollbackGate(gate_id="smoke_rollback", required_consecutive_failures=2),
    )
    assert rollback.decision == "rollback"
    print(
        {
            "datasets": len(matrix),
            "feature_coverage": sorted(coverage),
            "profile_store_domains": sorted(profiles),
            "profile_lifecycle_domains": sorted(lifecycle_artifacts),
            "promotion_decision": decision.decision,
            "positive_promotion_decision": promote.decision,
            "slice_hold_decision": slice_hold.decision,
            "multi_window_decision": multi_window_hold.decision,
            "previous_stable_required_decision": previous_stable_missing.decision,
            "rollback_decision": rollback.decision,
            "promotion_reasons": decision.reasons,
        }
    )


def _tiny_rolling_dataset() -> FootballBacktestDataset:
    records: list[FootballMatchRecord] = []
    for season_index, season in enumerate(["2122", "2223", "2324"]):
        for match_index in range(4):
            records.append(
                FootballMatchRecord(
                    match_id=f"smoke-{season}-{match_index}",
                    dataset_id="tiny_previous_stable_smoke",
                    competition="smoke_league",
                    season=season,
                    event_time=datetime.fromisoformat(f"20{20 + season_index}-08-{match_index + 1:02d}T15:00:00"),
                    home_team=f"Home {match_index}",
                    away_team=f"Away {match_index}",
                    home_goals=2 if match_index % 3 == 0 else 0,
                    away_goals=0 if match_index % 3 == 0 else 1,
                    odds={"home": 1.9, "draw": 3.4, "away": 4.2},
                    odds_snapshot_type="T-24h",
                )
            )
    return FootballBacktestDataset(
        manifest=DatasetManifest(
            dataset_id="tiny_previous_stable_smoke",
            source_url="local",
            license_note="Synthetic previous-stable smoke dataset.",
            competition="smoke_league",
            season="2122,2223,2324",
            case_count=len(records),
            horizon_policy="T-24h",
            snapshot_policy="synthetic_smoke",
        ),
        records=records,
    )


if __name__ == "__main__":
    main()
