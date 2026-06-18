import math

from app.evidence.store import InMemoryEvidenceStore
from app.feedback.ledger import PredictionLedger
from app.schemas import FeedbackMetrics, FeedbackRequest


class FeedbackAnalyzer:
    def __init__(self, store: InMemoryEvidenceStore, ledger: PredictionLedger | None = None):
        self.store = store
        self.ledger = ledger

    async def analyze(self, request: FeedbackRequest) -> dict[str, object]:
        self.store.add_feedback(request)
        prediction = self.store.get_prediction(request.task_id)
        if not prediction:
            ledger_prediction = self._ledger_prediction(request.task_id)
            if ledger_prediction:
                outcomes = ledger_prediction.get("outcomes")
                metrics = self._metrics(
                    {str(key): float(value) for key, value in outcomes.items()}
                    if isinstance(outcomes, dict)
                    else {},
                    request.actual_outcome,
                )
                return {
                    "task_id": request.task_id,
                    "actual_outcome": request.actual_outcome,
                    "metrics": metrics.model_dump(),
                    "prediction_source": "persistent_ledger",
                    "factor_gaps": ledger_prediction.get("missing_evidence", []),
                    "optimization": self._optimization(
                        metrics,
                        ledger_prediction.get("factor_summaries", []),
                        ledger_prediction.get("missing_evidence", []),
                    ),
                    "next_actions": [
                        "Generate an EvaluationReport row from this resolved ledger event.",
                        "Compare error attribution against current ensemble weights.",
                        "Propose a new weight profile only after enough resolved cases accumulate.",
                    ],
                }
            return {
                "task_id": request.task_id,
                "actual_outcome": request.actual_outcome,
                "metrics": FeedbackMetrics().model_dump(),
                "prediction_source": "missing",
                "next_actions": [
                    "Prediction snapshot was not found in memory.",
                    "Check the persistent ledger before relying on feedback metrics across restarts.",
                ],
            }

        metrics = self._metrics(prediction.outcomes, request.actual_outcome)
        factor_gaps = [
            factor.label
            for factor in prediction.factors
            if factor.evidence_count == 0 or factor.confidence < 0.45
        ][:5]

        return {
            "task_id": request.task_id,
            "actual_outcome": request.actual_outcome,
            "metrics": metrics.model_dump(),
            "prediction_source": "memory",
            "factor_gaps": factor_gaps,
            "optimization": self._optimization(
                metrics,
                [factor.model_dump() for factor in prediction.factors],
                factor_gaps,
            ),
            "next_actions": [
                "Collect post-event official reports and statistics.",
                "Compare pre-event evidence against resolved outcome.",
                "Update source reliability and factor weights in a new version.",
            ],
        }

    def _metrics(self, outcomes: dict[str, float], actual: str) -> FeedbackMetrics:
        if actual not in outcomes:
            return FeedbackMetrics(
                top_prediction=self._top_prediction(outcomes),
            )
        brier = sum((probability - (1 if outcome == actual else 0)) ** 2 for outcome, probability in outcomes.items())
        probability = max(outcomes[actual], 1e-9)
        top = self._top_prediction(outcomes)
        return FeedbackMetrics(
            brier_score=round(brier, 6),
            log_loss=round(-math.log(probability), 6),
            predicted_probability=round(outcomes[actual], 6),
            top_prediction=top,
            was_top_prediction_correct=(top == actual if top != "tie" else None),
        )

    def _ledger_prediction(self, task_id: str) -> dict[str, object] | None:
        if not self.ledger:
            return None
        summary = self.ledger.task_summary(task_id)
        latest = summary.get("latest_prediction")
        if not isinstance(latest, dict):
            return None
        response = latest.get("response")
        return response if isinstance(response, dict) else None

    def _top_prediction(self, outcomes: dict[str, float]) -> str | None:
        if not outcomes:
            return None
        highest = max(outcomes.values())
        winners = [outcome for outcome, probability in outcomes.items() if abs(probability - highest) < 1e-9]
        if len(winners) > 1:
            return "tie"
        return winners[0]

    def _optimization(
        self,
        metrics: FeedbackMetrics,
        factors: list[dict[str, object]],
        factor_gaps: list[object],
    ) -> dict[str, object]:
        gap_text = " ".join(str(item).lower() for item in factor_gaps)
        weak_factors = [
            str(factor.get("key") or factor.get("label"))
            for factor in factors
            if _float(factor.get("confidence")) < 0.45 or int(_float(factor.get("evidence_count"))) == 0
        ]
        recommended_data_work = []
        if "market" in gap_text or "odds" in gap_text or "market_odds" in weak_factors:
            recommended_data_work.append("Prioritize legal structured 1X2 odds before kickoff; web snippets are not enough.")
        if "lineup" in gap_text or "availability" in gap_text or "lineup_availability" in weak_factors:
            recommended_data_work.append("Add confirmed/predicted lineup and injury availability snapshots near T-24/T-2.")
        if "tactical" in gap_text or "tactical_matchup" in weak_factors:
            recommended_data_work.append("Extract directional tactical claims into structured support/oppose signals.")
        if "chemistry" in gap_text or "relationships" in gap_text:
            recommended_data_work.append("Treat chemistry/relationship evidence as low-priority unless it is recent and corroborated.")

        weight_action = "hold"
        if metrics.was_top_prediction_correct is False:
            weight_action = "review_after_batch"
        if metrics.top_prediction == "tie":
            weight_action = "increase_directional_data_before_weight_change"

        return {
            "error_level": self._error_level(metrics),
            "weight_action": weight_action,
            "recommended_data_work": recommended_data_work[:5],
            "release_gate": (
                "Do not publish a new weight profile from one match; accumulate a resolved batch and compare Brier/log-loss."
            ),
        }

    def _error_level(self, metrics: FeedbackMetrics) -> str:
        if metrics.brier_score is None:
            return "unknown"
        if metrics.was_top_prediction_correct is True and metrics.brier_score < 0.55:
            return "acceptable_single_case"
        if metrics.top_prediction == "tie":
            return "underconfident_no_direction"
        if metrics.was_top_prediction_correct is False:
            return "miss"
        return "needs_review"


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
