from app.schemas import EvidenceItem, FeedbackRequest, PredictionResponse


class InMemoryEvidenceStore:
    def __init__(self):
        self._items: dict[str, list[EvidenceItem]] = {}
        self._predictions: dict[str, PredictionResponse] = {}
        self._feedback: dict[str, list[FeedbackRequest]] = {}

    def add_many(self, task_id: str, items: list[EvidenceItem]) -> None:
        self._items.setdefault(task_id, []).extend(items)

    def list_for_task(self, task_id: str) -> list[EvidenceItem]:
        return self._items.get(task_id, [])

    def save_prediction(self, prediction: PredictionResponse) -> None:
        self._predictions[prediction.task_id] = prediction

    def get_prediction(self, task_id: str) -> PredictionResponse | None:
        return self._predictions.get(task_id)

    def add_feedback(self, feedback: FeedbackRequest) -> None:
        self._feedback.setdefault(feedback.task_id, []).append(feedback)

    def list_feedback(self, task_id: str) -> list[FeedbackRequest]:
        return self._feedback.get(task_id, [])
