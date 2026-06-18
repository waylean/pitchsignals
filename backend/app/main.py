from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import date

from app.data_sources.football_schedule import FootballScheduleService
from app.domain_packs.registry import list_domain_packs
from app.schemas import FeedbackRequest, FootballScheduleResponse, PredictionRequest, PredictionResponse
from app.workflows.orchestrator import PredictionWorkflow

app = FastAPI(title="Football Match Predictor", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
workflow = PredictionWorkflow()
football_schedule_service = FootballScheduleService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/domain-packs")
def domain_packs():
    return list_domain_packs()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    return await workflow.predict(request)


@app.get("/football/schedule/today", response_model=FootballScheduleResponse)
async def football_schedule_today(
    date_: date | None = Query(default=None, alias="date"),
    timezone: str = "Asia/Shanghai",
    days: int = 2,
) -> FootballScheduleResponse:
    return await football_schedule_service.today(date_, timezone, days)


@app.post("/feedback")
async def feedback(request: FeedbackRequest) -> dict[str, object]:
    return await workflow.feedback(request)


@app.get("/ledger")
def ledger(limit: int = 50) -> dict[str, object]:
    return {
        "status": "ready",
        "events": workflow.ledger_events(limit=max(min(limit, 500), 0)),
    }


@app.get("/ledger/{task_id}")
def ledger_task(task_id: str) -> dict[str, object]:
    return workflow.ledger_task_summary(task_id)


@app.get("/backtests/worldcup-2022/group-stage/latest")
def worldcup_2022_group_backtest() -> dict[str, object]:
    path = Path(__file__).resolve().parents[2] / "outputs" / "worldcup_2022_group_backtest.json"
    if not path.exists():
        return {
            "status": "missing",
            "reason": "Run backend/scripts/football_worldcup_backtest.py first.",
        }
    import json

    return {"status": "ready", "data": json.loads(path.read_text(encoding="utf-8"))}
