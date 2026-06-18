import asyncio
from datetime import datetime
from hashlib import sha1
import json
from pathlib import Path

from app.core.text import expand_query
from app.data_sources.base import EvidenceQuery
from app.data_sources.football_data_fixtures_odds import FootballDataFixturesOddsSource
from app.data_sources.football_data_odds import FootballDataOddsSource
from app.data_sources.football_match_context import FootballMatchContextSource
from app.data_sources.football_odds_snapshot_csv import FootballOddsSnapshotCsvSource
from app.data_sources.football_public import FootballPublicSource
from app.data_sources.football_schedule_context import FootballScheduleContextSource
from app.data_sources.open_meteo import OpenMeteoWeatherSource
from app.data_sources.the_odds_api import TheOddsApiOddsSource
from app.data_sources.web_search import WebSearchSource
from app.data_sources.world_football_elo import WorldFootballEloSource
from app.domain_packs.registry import get_domain_pack
from app.evidence.scoring import score_recency
from app.evidence.store import InMemoryEvidenceStore
from app.evidence.verifier import EvidenceVerifier
from app.feedback.analyzer import FeedbackAnalyzer
from app.feedback.ledger import PredictionLedger
from app.core.config import settings
from app.models.research_review import LLMResearchReviewer
from app.models.football_horizon_profiles import apply_football_horizon_profile
from app.models.football_math import (
    FootballDixonColesModel,
    FootballEloStrengthModel,
    FootballMarketModel,
    FootballPoissonModel,
)
from app.models.governance import (
    ModelRegistry,
    PredictionDistribution,
    blend_distributions,
    distribution_agreement,
    select_ensemble_profile,
)
from app.models.question_decomposer import LLMQuestionDecomposer
from app.models.unofficial_signal_structurer import UnofficialFootballSignalStructurer
from app.models.weighted_model import WeightedPredictionModel
from app.schemas import EvidenceItem, FeedbackRequest, PredictionRequest, PredictionResponse


class PredictionWorkflow:
    def __init__(self):
        self.sources = [
            FootballScheduleContextSource(),
            WorldFootballEloSource(),
            FootballOddsSnapshotCsvSource(),
            FootballDataFixturesOddsSource(),
            FootballDataOddsSource(),
            TheOddsApiOddsSource(),
            FootballMatchContextSource(),
            OpenMeteoWeatherSource(),
            WebSearchSource(),
            FootballPublicSource(),
        ]
        self.evidence_store = InMemoryEvidenceStore()
        self.verifier = EvidenceVerifier()
        self.model = WeightedPredictionModel()
        self.model_registry = ModelRegistry()
        self.model_registry.register(FootballMarketModel())
        self.model_registry.register(FootballEloStrengthModel())
        self.model_registry.register(FootballPoissonModel())
        self.model_registry.register(FootballDixonColesModel())
        self.question_decomposer = LLMQuestionDecomposer()
        self.unofficial_structurer = UnofficialFootballSignalStructurer()
        self.research_reviewer = LLMResearchReviewer()
        self.ledger = PredictionLedger(settings.prediction_ledger_path)
        self.feedback_analyzer = FeedbackAnalyzer(self.evidence_store, self.ledger)

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        pack = get_domain_pack(request.domain)
        request = await self.question_decomposer.decompose(request, pack)
        task_id = self._task_id(request)
        horizon_profile = self._apply_horizon_profile(request, pack.key)
        active_weight_profile = self._apply_latest_weight_profile(request, pack.key)
        ensemble_profile = select_ensemble_profile(pack.key, request.context)
        trace = [
            "task_intake",
            f"domain_pack:{pack.key}",
            f"ai_question_decomposition:{(request.context.get('ai_decomposition') or {}).get('status', 'not_available')}",
            "factor_decomposition",
            *([f"horizon_profile:{horizon_profile}"] if horizon_profile else []),
            *([f"weight_profile:{active_weight_profile}"] if active_weight_profile else []),
            *([f"ensemble_profile:{ensemble_profile.profile_id}@{ensemble_profile.version}"] if ensemble_profile else []),
            "evidence_collection",
            "ai_unofficial_signal_structuring",
            "evidence_verification",
            "factor_scoring",
            "probability_estimation",
            "llm_research_review",
        ]

        queries = self._build_queries(request, pack)
        evidence = await self._collect_evidence(request, queries)
        evidence.extend(await self.unofficial_structurer.structure(request, evidence))
        evidence = await self.verifier.verify(evidence, request.prediction_deadline)
        self._apply_factor_freshness(evidence, pack)

        self.evidence_store.add_many(task_id, evidence)
        factors = self.model.score_factors(pack, request, evidence)
        primary_outcomes = self.model.predict(request, factors)
        model_runs = self.model_registry.run(request, pack, evidence, factors)
        auxiliary_distributions = [run.distribution for run in model_runs if run.status == "ok"]
        outcomes = blend_distributions(
            primary_outcomes,
            auxiliary_distributions,
            auxiliary_weight=0.22,
            ensemble_profile=ensemble_profile,
        )
        outcomes = self._apply_football_market_postprocess(request, pack.key, outcomes)

        factor_evidence = [item for item in evidence if item.impact_area != "general_context"]
        coverage = min(sum(f.evidence_count for f in factors) / max(len(factors) * 3, 1), 1)
        freshness = (
            round(sum(item.recency_score for item in factor_evidence) / len(factor_evidence), 4)
            if factor_evidence
            else 0
        )
        model_agreement = distribution_agreement(
            [
                PredictionDistribution(outcomes=primary_outcomes, model_id="weighted_evidence"),
                *auxiliary_distributions,
            ]
        )
        pick, pick_probability = self._top_outcome(outcomes)
        missing_evidence = self._missing_evidence(evidence, factors)
        data_completeness = self._data_completeness(coverage, missing_evidence)
        evidence_gate_status = self._evidence_gate_status(
            data_completeness,
            freshness,
            model_agreement,
            missing_evidence,
        )
        has_direction = self.model.has_directional_signal(factors)
        model_status = "evidence_directional" if has_direction else "evidence_neutral"
        confidence = round((coverage * 0.35 + freshness * 0.25 + model_agreement * 0.2), 4)
        if not has_direction:
            confidence = min(confidence, 0.48)
        research_review = await self.research_reviewer.review(request, pack, evidence, factors, outcomes)

        response = PredictionResponse(
            task_id=task_id,
            domain=pack.key,
            normalized_question=str(request.context.get("normalized_question") or request.question).strip(),
            outcomes=outcomes,
            pick=pick,
            pick_probability=pick_probability,
            model_status=model_status,
            confidence=confidence,
            data_coverage=round(coverage, 4),
            data_completeness=data_completeness,
            freshness=freshness,
            model_agreement=model_agreement,
            horizon_profile=horizon_profile,
            evidence_gate_status=evidence_gate_status,
            missing_evidence=missing_evidence,
            factors=factors,
            evidence=evidence,
            uncertainties=self._uncertainties(evidence, has_direction),
            workflow_trace=trace,
            research_review=research_review,
            model_runs=[
                {
                    "model_id": run.model_id,
                    "status": run.status,
                    "outcomes": run.distribution.normalized().outcomes,
                    "rationale": run.distribution.rationale,
                    "notes": run.notes or [],
                }
                for run in model_runs
            ],
            distribution_metrics={
                "primary_model": "weighted_evidence",
                "primary_entropy": PredictionDistribution(outcomes=primary_outcomes, model_id="weighted_evidence").entropy(),
                "model_agreement": model_agreement,
                "auxiliary_model_count": len(auxiliary_distributions),
                "ensemble_profile": (
                    {
                        "profile_id": ensemble_profile.profile_id,
                        "version": ensemble_profile.version,
                        "maturity": ensemble_profile.maturity,
                        "horizon_profile": ensemble_profile.horizon_profile,
                        "model_weights": ensemble_profile.model_weights,
                    }
                    if ensemble_profile
                    else None
                ),
            },
        )
        self.evidence_store.save_prediction(response)
        self.ledger.record_prediction(request, response)
        return response

    async def feedback(self, request: FeedbackRequest) -> dict[str, object]:
        result = await self.feedback_analyzer.analyze(request)
        self.ledger.record_feedback(request, result)
        return result

    def ledger_events(self, task_id: str | None = None, limit: int | None = None) -> list[dict[str, object]]:
        return [
            {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "task_id": event.task_id,
                "recorded_at": event.recorded_at,
                "payload": event.payload,
            }
            for event in self.ledger.events(task_id=task_id, limit=limit)
        ]

    def ledger_task_summary(self, task_id: str) -> dict[str, object]:
        return self.ledger.task_summary(task_id)

    def _build_queries(self, request: PredictionRequest, pack) -> list[EvidenceQuery]:
        base = self._query_base(request, pack)
        queries = [
            EvidenceQuery(
                text=f"{base} prediction {pack.label}",
                impact_area="general_context",
                factor_label="General Context",
            )
        ]
        for factor in pack.factors:
            hints = " ".join(factor.evidence_queries[:2])
            if hints:
                queries.append(
                    EvidenceQuery(
                        text=f"{base} {hints}",
                        impact_area=factor.key,
                        factor_label=factor.label,
                    )
                )
        return queries

    def _task_id(self, request: PredictionRequest) -> str:
        raw = f"{request.domain}|{request.question}|{request.event_time}|{request.outcome_type}"
        return sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _query_base(self, request: PredictionRequest, pack) -> str:
        search_query = request.context.get("search_query")
        if isinstance(search_query, str) and search_query.strip():
            return search_query.strip()
        competitors = request.context.get("competitors")
        if isinstance(competitors, list) and len(competitors) >= 2:
            joined = " ".join(str(item) for item in competitors[:2])
            return f"{joined} {pack.label}"
        return expand_query(request.question.strip())

    def _uncertainties(self, evidence, has_direction: bool) -> list[str]:
        notes = [
            "Collected web results are candidate evidence, not fully verified facts.",
        ]
        if not has_direction:
            notes.append("No strong directional signal was extracted, so probabilities remain near baseline.")
        if any(item.evidence_stage == "collection_error" for item in evidence):
            notes.append("Some collection attempts failed and were excluded from factor scoring.")
        if any(item.evidence_stage == "collection_gap" for item in evidence):
            notes.append("Some requested data was unavailable because required event details or public sources were missing.")
        return notes

    def _top_outcome(self, outcomes: dict[str, float]) -> tuple[str | None, float | None]:
        if not outcomes:
            return None, None
        top_outcome, top_probability = max(outcomes.items(), key=lambda item: item[1])
        return top_outcome, round(top_probability, 6)

    def _apply_football_market_postprocess(
        self,
        request: PredictionRequest,
        domain_key: str,
        outcomes: dict[str, float],
    ) -> dict[str, float]:
        if domain_key != "football" or len(outcomes) < 3:
            return outcomes
        schedule_odds = request.context.get("schedule_odds")
        if not isinstance(schedule_odds, dict) or not schedule_odds.get("available"):
            return outcomes
        keys = request.outcomes or list(outcomes)
        if len(keys) < 3:
            return outcomes
        try:
            home_odds = float(schedule_odds["home_odds"])
            draw_odds = float(schedule_odds["draw_odds"])
            away_odds = float(schedule_odds["away_odds"])
            bookmaker_count = int(schedule_odds.get("bookmaker_count") or 0)
        except (KeyError, TypeError, ValueError):
            return outcomes
        if min(home_odds, draw_odds, away_odds) <= 1:
            return outcomes

        raw = {
            keys[0]: 1 / home_odds,
            keys[1]: 1 / draw_odds,
            keys[2]: 1 / away_odds,
        }
        total = sum(raw.values()) or 1
        market = {key: value / total for key, value in raw.items()}
        reliability = min(0.48 + bookmaker_count / 100, 0.68)
        blended = {
            key: outcomes.get(key, 0) * (1 - reliability) + market.get(key, 0) * reliability
            for key in keys[:3]
        }

        favorite_key, favorite_prob = max(market.items(), key=lambda item: item[1])
        longshot_key, longshot_prob = min(
            ((key, value) for key, value in market.items() if key != keys[1]),
            key=lambda item: item[1],
        )
        extreme = favorite_prob >= 0.68 or longshot_prob <= 0.1 or max(home_odds, away_odds) >= 8
        if extreme:
            # Favorite-longshot markets can overstate extreme favorites. Keep the
            # prediction decisive, but reserve upset/draw mass instead of blindly
            # following the shortest price.
            favorite_tax = min(max((favorite_prob - 0.62) * 0.45, 0.035), 0.095)
            available_tax = min(favorite_tax, max(blended.get(favorite_key, 0) - 0.42, 0))
            if available_tax > 0:
                blended[favorite_key] -= available_tax
                blended[keys[1]] = blended.get(keys[1], 0) + available_tax * 0.62
                blended[longshot_key] = blended.get(longshot_key, 0) + available_tax * 0.38

        total = sum(max(value, 0) for value in blended.values()) or 1
        return {key: round(max(value, 0) / total, 4) for key, value in blended.items()}

    def _missing_evidence(self, evidence, factors) -> list[str]:
        missing: list[str] = []
        for item in evidence:
            if item.evidence_stage == "collection_gap":
                missing.append(f"{item.impact_area}: {item.claim}")
            elif item.evidence_stage == "collection_error":
                missing.append(f"{item.impact_area}: collection error from {item.source}")
            elif not item.deadline_compliant:
                missing.append(f"{item.impact_area}: evidence missed prediction deadline")
        for factor in factors:
            if factor.evidence_count == 0:
                missing.append(f"{factor.key}: no usable evidence")
            elif factor.confidence < 0.45:
                missing.append(f"{factor.key}: weak evidence confidence {factor.confidence:.2f}")
        return list(dict.fromkeys(missing))[:20]

    def _data_completeness(self, coverage: float, missing_evidence: list[str]) -> float:
        penalty = min(len(missing_evidence) * 0.035, 0.35)
        return round(max(min(coverage - penalty, 1), 0), 4)

    def _evidence_gate_status(
        self,
        data_completeness: float,
        freshness: float,
        model_agreement: float,
        missing_evidence: list[str],
    ) -> str:
        if data_completeness >= 0.8 and freshness >= 0.65 and model_agreement >= 0.55 and not missing_evidence:
            return "pass"
        if data_completeness >= 0.65 and freshness >= 0.5:
            return "warn"
        if data_completeness >= 0.4:
            return "weak"
        return "blocked"

    def _apply_latest_weight_profile(self, request: PredictionRequest, domain_key: str) -> str | None:
        if domain_key != "football" or isinstance(request.context.get("weight_profile"), dict):
            return None
        profile, profile_id, mature = self._latest_football_weight_profile()
        if not profile:
            return None
        if not mature and not request.context.get("use_experimental_weight_profile"):
            request.context["weight_profile_skipped"] = profile_id
            request.context["weight_profile_skip_reason"] = "insufficient_directional_coverage"
            return None
        request.context["weight_profile"] = profile
        request.context["weight_profile_source"] = profile_id
        return profile_id

    def _apply_horizon_profile(self, request: PredictionRequest, domain_key: str) -> str | None:
        if domain_key != "football":
            return None
        return apply_football_horizon_profile(
            request.context,
            request.event_time,
            request.prediction_deadline,
        )

    def _latest_football_weight_profile(self) -> tuple[dict[str, float] | None, str | None, bool]:
        path = Path(__file__).resolve().parents[3] / "outputs" / "worldcup_2022_group_backtest.json"
        if not path.exists():
            return None, None, False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None, None, False
        weights = payload.get("trained_weights")
        if not isinstance(weights, dict):
            return None, None, False

        coverage = payload.get("directional_coverage") or {}
        match_count = int((payload.get("metrics") or {}).get("matches") or 0)
        covered_factors = sum(
            1
            for value in coverage.values()
            if isinstance(value, (int, float)) and match_count and value >= max(12, match_count * 0.25)
        )
        profile_id = (
            "worldcup_2022_group_stage"
            if covered_factors >= 3
            else "calibration_sparse_worldcup_2022_group_stage"
        )
        return weights, profile_id, covered_factors >= 3

    def _apply_factor_freshness(self, evidence, pack) -> None:
        half_lives = {factor.key: factor.half_life_hours for factor in pack.factors}
        for item in evidence:
            if item.evidence_stage in {"collection_error", "collection_gap"}:
                continue
            half_life = half_lives.get(item.impact_area)
            if half_life and item.published_at:
                item.recency_score = score_recency(item.published_at, half_life)

    async def _collect_evidence(
        self,
        request: PredictionRequest,
        queries: list[EvidenceQuery],
    ) -> list[EvidenceItem]:
        batches = await asyncio.gather(
            *(self._collect_from_source(source, request, queries) for source in self.sources),
            return_exceptions=True,
        )
        evidence: list[EvidenceItem] = []
        for source, batch in zip(self.sources, batches):
            if isinstance(batch, Exception):
                evidence.append(
                    EvidenceItem(
                        claim=f"{source.name} collection failed: {batch!r}",
                        source=source.name,
                        evidence_stage="collection_error",
                        collected_at=datetime.utcnow(),
                        impact_area="collection_error",
                        source_reliability=0.25,
                        recency_score=0.0,
                        confidence=0.0,
                    )
                )
                continue
            evidence.extend(batch)
        return evidence

    async def _collect_from_source(
        self,
        source,
        request: PredictionRequest,
        queries: list[EvidenceQuery],
    ) -> list[EvidenceItem]:
        return await source.collect(request, queries)
