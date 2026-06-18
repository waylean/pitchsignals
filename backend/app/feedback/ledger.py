from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha1
import json
from pathlib import Path
from typing import Any

from app.schemas import FeedbackRequest, PredictionRequest, PredictionResponse


@dataclass(frozen=True)
class PredictionLedgerEvent:
    event_id: str
    event_type: str
    task_id: str
    recorded_at: str
    payload: dict[str, Any]


class PredictionLedger:
    def __init__(self, path: Path | str):
        self.path = _resolve_path(path)

    def record_prediction(
        self,
        request: PredictionRequest,
        response: PredictionResponse,
    ) -> PredictionLedgerEvent:
        payload = {
            "request": request.model_dump(mode="json"),
            "response": _prediction_summary(response),
        }
        return self.append("prediction_created", response.task_id, payload)

    def record_feedback(
        self,
        request: FeedbackRequest,
        analysis: dict[str, Any],
    ) -> PredictionLedgerEvent:
        payload = {
            "feedback": request.model_dump(mode="json"),
            "analysis": _json_safe(analysis),
        }
        return self.append("outcome_resolved", request.task_id, payload)

    def append(
        self,
        event_type: str,
        task_id: str,
        payload: dict[str, Any],
    ) -> PredictionLedgerEvent:
        recorded_at = datetime.utcnow().isoformat()
        event_id = _event_id(event_type, task_id, recorded_at, payload)
        event = PredictionLedgerEvent(
            event_id=event_id,
            event_type=event_type,
            task_id=task_id,
            recorded_at=recorded_at,
            payload=payload,
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_event_to_dict(event), ensure_ascii=False, sort_keys=True))
            file.write("\n")
        return event

    def events(self, task_id: str | None = None, limit: int | None = None) -> list[PredictionLedgerEvent]:
        if not self.path.exists():
            return []
        rows: list[PredictionLedgerEvent] = []
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = _event_from_dict(json.loads(line))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if task_id and event.task_id != task_id:
                    continue
                rows.append(event)
        if limit is not None and limit >= 0:
            return rows[-limit:]
        return rows

    def task_summary(self, task_id: str) -> dict[str, Any]:
        events = self.events(task_id=task_id)
        latest_prediction = next(
            (event for event in reversed(events) if event.event_type == "prediction_created"),
            None,
        )
        latest_resolution = next(
            (event for event in reversed(events) if event.event_type == "outcome_resolved"),
            None,
        )
        return {
            "task_id": task_id,
            "event_count": len(events),
            "has_prediction": latest_prediction is not None,
            "has_resolution": latest_resolution is not None,
            "latest_prediction": latest_prediction.payload if latest_prediction else None,
            "latest_resolution": latest_resolution.payload if latest_resolution else None,
            "events": [_event_to_dict(event) for event in events],
        }


def _prediction_summary(response: PredictionResponse) -> dict[str, Any]:
    top_outcome, top_probability = _top_outcome(response.outcomes)
    top_outcome = response.pick or top_outcome
    top_probability = response.pick_probability if response.pick_probability is not None else top_probability
    evidence_snapshot_ids = [
        item.evidence_id
        for item in response.evidence
        if item.evidence_id
    ]
    evidence_sources = sorted(
        {
            str(item.source_url or item.source)
            for item in response.evidence
            if item.source_url or item.source
        }
    )[:50]
    return {
        "task_id": response.task_id,
        "domain": response.domain,
        "normalized_question": response.normalized_question,
        "outcomes": response.outcomes,
        "pick": top_outcome,
        "pick_probability": top_probability,
        "confidence": response.confidence,
        "data_coverage": response.data_coverage,
        "data_completeness": response.data_completeness,
        "freshness": response.freshness,
        "model_agreement": response.model_agreement,
        "horizon_profile": response.horizon_profile,
        "evidence_gate_status": response.evidence_gate_status,
        "missing_evidence": response.missing_evidence,
        "model_status": response.model_status,
        "uncertainties": response.uncertainties,
        "workflow_trace": response.workflow_trace,
        "distribution_metrics": response.distribution_metrics,
        "model_runs": response.model_runs,
        "factor_summaries": [
            {
                "key": factor.key,
                "label": factor.label,
                "value": factor.value,
                "weight": factor.weight,
                "confidence": factor.confidence,
                "evidence_count": factor.evidence_count,
            }
            for factor in response.factors
        ],
        "evidence_count": len(response.evidence),
        "evidence_snapshot_ids": evidence_snapshot_ids[:100],
        "evidence_sources": evidence_sources,
    }


def _event_id(
    event_type: str,
    task_id: str,
    recorded_at: str,
    payload: dict[str, Any],
) -> str:
    raw = json.dumps(
        {
            "event_type": event_type,
            "task_id": task_id,
            "recorded_at": recorded_at,
            "payload": _json_safe(payload),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return sha1(raw.encode("utf-8")).hexdigest()[:20]


def _top_outcome(outcomes: dict[str, float]) -> tuple[str | None, float | None]:
    if not outcomes:
        return None, None
    top = max(outcomes.items(), key=lambda item: item[1])
    return top[0], round(top[1], 6)


def _event_to_dict(event: PredictionLedgerEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "task_id": event.task_id,
        "recorded_at": event.recorded_at,
        "payload": _json_safe(event.payload),
    }


def _event_from_dict(raw: dict[str, Any]) -> PredictionLedgerEvent:
    return PredictionLedgerEvent(
        event_id=str(raw["event_id"]),
        event_type=str(raw["event_type"]),
        task_id=str(raw["task_id"]),
        recorded_at=str(raw["recorded_at"]),
        payload=raw.get("payload") if isinstance(raw.get("payload"), dict) else {},
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _resolve_path(path: Path | str) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    root = Path(__file__).resolve().parents[3]
    return root / path
