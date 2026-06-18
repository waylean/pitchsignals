from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from app.models.governance import (
    EnsembleProfile,
    ProfileLifecycleArtifact,
    PromotionDecision,
    RollbackDecision,
    ensemble_profile_from_dict,
)


class ProfileLifecycleStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    def promote_to_stable(
        self,
        profile: EnsembleProfile,
        decision: PromotionDecision,
        source_report_id: str | None = None,
        previous_stable: EnsembleProfile | None = None,
        artifacts: dict[str, str] | None = None,
        allow_market_proxy_stable: bool = False,
    ) -> ProfileLifecycleArtifact:
        if decision.decision != "promote":
            raise ValueError("stable promotion requires a promote decision")
        if profile.horizon_profile == "market_snapshot_unknown_or_closing_proxy" and not allow_market_proxy_stable:
            raise ValueError("market-proxy profiles cannot become stable without explicit override")

        stable_profile = self._stable_profile(profile)
        profile_path = self.profile_path(stable_profile)
        self._write_json(profile_path, _profile_to_dict(stable_profile))
        pointer_path = self.stable_pointer_path(stable_profile.domain)
        self._write_json(
            pointer_path,
            {
                "domain": stable_profile.domain,
                "profile_id": stable_profile.profile_id,
                "version": stable_profile.version,
                "profile_path": str(profile_path.relative_to(self.root)),
                "promoted_at": stable_profile.promoted_at,
                "source_report_id": source_report_id,
                "previous_stable_profile_id": previous_stable.profile_id if previous_stable else None,
                "previous_stable_version": previous_stable.version if previous_stable else None,
            },
        )
        lifecycle = ProfileLifecycleArtifact(
            artifact_id=f"{stable_profile.profile_id}@{stable_profile.version}_stable",
            profile_id=stable_profile.profile_id,
            profile_version=stable_profile.version,
            domain=stable_profile.domain,
            status="stable",
            decision=asdict(decision),
            source_report_id=source_report_id,
            created_at=stable_profile.promoted_at,
            previous_profile_version=previous_stable.version if previous_stable else None,
            artifacts={
                "profile": str(profile_path.relative_to(self.root)),
                "stable_pointer": str(pointer_path.relative_to(self.root)),
                **(artifacts or {}),
            },
            metrics=decision.metrics,
            next_actions=[
                "Monitor the stable profile against rollback gates after each resolved evaluation window.",
                "Create a rollback lifecycle artifact if consecutive evaluation windows degrade.",
            ],
        )
        self._write_lifecycle(lifecycle)
        return lifecycle

    def record_rollback(
        self,
        domain: str,
        decision: RollbackDecision,
        rollback_target: EnsembleProfile | None = None,
        source_report_id: str | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> ProfileLifecycleArtifact:
        if decision.decision != "rollback":
            raise ValueError("rollback artifact requires a rollback decision")
        created_at = datetime.utcnow().isoformat()
        pointer_path = self.stable_pointer_path(domain)
        if rollback_target:
            target_path = self.profile_path(rollback_target)
            self._write_json(target_path, _profile_to_dict(rollback_target))
            self._write_json(
                pointer_path,
                {
                    "domain": rollback_target.domain,
                    "profile_id": rollback_target.profile_id,
                    "version": rollback_target.version,
                    "profile_path": str(target_path.relative_to(self.root)),
                    "promoted_at": rollback_target.promoted_at,
                    "rollback_at": created_at,
                    "rollback_reason": decision.reasons,
                    "source_report_id": source_report_id,
                },
            )
        lifecycle = ProfileLifecycleArtifact(
            artifact_id=f"{decision.stable_model_id}_rollback_{created_at.replace(':', '').replace('.', '')}",
            profile_id=decision.stable_model_id,
            profile_version=str((decision.metrics or {}).get("stable_version") or "unknown"),
            domain=domain,
            status="rolled_back",
            decision=asdict(decision),
            source_report_id=source_report_id,
            created_at=created_at,
            previous_profile_version=rollback_target.version if rollback_target else None,
            artifacts={
                "stable_pointer": str(pointer_path.relative_to(self.root)),
                **(artifacts or {}),
            },
            metrics=decision.metrics,
            next_actions=[
                "Inspect the failing windows before allowing the rolled-back profile to be promoted again.",
                "Require a fresh promotion decision against the restored stable baseline.",
            ],
        )
        self._write_lifecycle(lifecycle)
        return lifecycle

    def load_stable_pointer(self, domain: str) -> dict[str, Any] | None:
        path = self.stable_pointer_path(domain)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return raw if isinstance(raw, dict) else None

    def load_stable_profile(self, domain: str) -> EnsembleProfile | None:
        pointer = self.load_stable_pointer(domain)
        if not pointer:
            return None
        profile_path = pointer.get("profile_path")
        if not profile_path:
            return None
        path = self.root / str(profile_path)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return ensemble_profile_from_dict(raw, fallback_domain=domain)
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def profile_path(self, profile: EnsembleProfile) -> Path:
        return self.root / profile.domain / f"{profile.profile_id}@{profile.version}.json"

    def stable_pointer_path(self, domain: str) -> Path:
        return self.root / domain / "stable.json"

    def _write_lifecycle(self, artifact: ProfileLifecycleArtifact) -> None:
        path = self.root / artifact.domain / "lifecycle" / f"{artifact.artifact_id}.lifecycle.json"
        self._write_json(path, asdict(artifact))

    def _stable_profile(self, profile: EnsembleProfile) -> EnsembleProfile:
        now = datetime.utcnow().isoformat()
        return EnsembleProfile(
            profile_id=profile.profile_id,
            version=profile.version,
            domain=profile.domain,
            model_weights=profile.model_weights,
            maturity="stable",
            horizon_profile=profile.horizon_profile,
            competition_scope=profile.competition_scope,
            data_tier=profile.data_tier,
            training_dataset_id=profile.training_dataset_id,
            validation_report_id=profile.validation_report_id,
            test_report_id=profile.test_report_id,
            lineage={
                **profile.lineage,
                "stable_writeback_at": now,
                "previous_maturity": profile.maturity,
            },
            created_at=profile.created_at,
            promoted_at=now,
            metrics=profile.metrics,
            notes=[
                *profile.notes,
                "Promoted to stable by ProfileLifecycleStore after passing promotion gates.",
            ],
        )

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _profile_to_dict(profile: EnsembleProfile) -> dict[str, Any]:
    raw = asdict(profile)
    return {key: value for key, value in raw.items() if value is not None}
