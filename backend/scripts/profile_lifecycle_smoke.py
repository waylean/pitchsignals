from __future__ import annotations

from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.models.governance import (  # noqa: E402
    EnsembleProfile,
    PromotionDecision,
    RollbackDecision,
)
from app.models.profile_lifecycle import ProfileLifecycleStore  # noqa: E402


def main() -> None:
    root = ROOT / "work" / "profile_lifecycle_smoke"
    if root.exists():
        shutil.rmtree(root)
    store = ProfileLifecycleStore(root)
    previous = EnsembleProfile(
        profile_id="football_stable_baseline",
        version="1.0.0",
        domain="football",
        maturity="stable",
        horizon_profile="T-24h",
        model_weights={"weighted_evidence": 1.0},
        promoted_at="2026-01-01T00:00:00",
    )
    candidate = EnsembleProfile(
        profile_id="football_horizon_candidate",
        version="1.1.0",
        domain="football",
        maturity="promoted",
        horizon_profile="T-24h",
        data_tier="public_free_horizon_safe",
        model_weights={"weighted_evidence": 0.4, "football_market_only": 0.6},
        metrics={"test_cases": 1200, "log_loss": 0.91},
    )
    decision = PromotionDecision(
        gate_id="football_1_0_horizon_gate",
        candidate_model_id="football_horizon_candidate",
        baseline_model_id="previous_stable",
        decision="promote",
        reasons=["candidate passed previous-stable and raw-market gates"],
        metrics={"test_cases": 1200, "log_loss_delta": -0.025},
    )
    lifecycle = store.promote_to_stable(
        candidate,
        decision,
        source_report_id="horizon_report",
        previous_stable=previous,
    )
    assert lifecycle.status == "stable"
    assert store.stable_pointer_path("football").exists()
    stable = store.load_stable_profile("football")
    assert stable is not None
    assert stable.maturity == "stable"
    assert stable.profile_id == "football_horizon_candidate"

    market_proxy = EnsembleProfile(
        profile_id="football_market_proxy_candidate",
        version="0.7.1",
        domain="football",
        maturity="promoted",
        horizon_profile="market_snapshot_unknown_or_closing_proxy",
        model_weights={"football_data_market_league_calibrated": 1.0},
    )
    try:
        store.promote_to_stable(market_proxy, decision)
    except ValueError as exc:
        assert "market-proxy" in str(exc)
    else:
        raise AssertionError("market proxy profile should not become stable by default")

    rollback = RollbackDecision(
        gate_id="football_rollback_gate",
        stable_model_id="football_horizon_candidate",
        baseline_model_id="raw_market",
        decision="rollback",
        reasons=["2 consecutive windows failed"],
        metrics={"stable_version": "1.1.0", "consecutive_failures": 2},
    )
    rollback_lifecycle = store.record_rollback(
        "football",
        rollback,
        rollback_target=previous,
        source_report_id="rollback_report",
    )
    assert rollback_lifecycle.status == "rolled_back"
    restored = store.load_stable_profile("football")
    assert restored is not None
    assert restored.profile_id == previous.profile_id
    print(
        {
            "stable_lifecycle": lifecycle.artifact_id,
            "rollback_lifecycle": rollback_lifecycle.artifact_id,
            "stable_pointer": str(store.stable_pointer_path("football")),
        }
    )


if __name__ == "__main__":
    main()
