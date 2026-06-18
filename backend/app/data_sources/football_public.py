from app.data_sources.base import EvidenceQuery
from app.data_sources.web_search import WebSearchSource
from app.schemas import EvidenceItem, PredictionRequest


class FootballPublicSource:
    """Focused public/free football collector.

    This source does not require paid APIs or API keys. It uses public search
    queries aimed at official, local-media, and clearly marked unofficial
    football references. Unofficial items remain candidate evidence only.
    """

    name = "football_public_search"

    def __init__(self):
        self.web_search = WebSearchSource(max_queries=14, results_per_query=3)

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if request.domain != "football":
            return []

        focused_queries = self._build_focused_queries(request)
        items = await self.web_search.collect(request, focused_queries)
        for item in items:
            item.source = self.name
        return items

    def _build_focused_queries(self, request: PredictionRequest) -> list[EvidenceQuery]:
        base = self._query_base(request)
        return [
            EvidenceQuery(
                text=f"{base} FIFA World Cup 2026 fixture preview",
                impact_area="general_context",
                factor_label="Official Context",
            ),
            EvidenceQuery(
                text=f"{base} odds 1X2 decimal odds Oddspedia Oddschecker World Cup 2026",
                impact_area="market_odds",
                factor_label="Market Odds",
            ),
            EvidenceQuery(
                text=f"{base} betting odds home draw away latest",
                impact_area="market_odds",
                factor_label="Market Odds",
            ),
            EvidenceQuery(
                text=f"{base} World Football Elo FIFA ranking",
                impact_area="team_strength",
                factor_label="Team Strength",
            ),
            EvidenceQuery(
                text=f"{base} team news injuries suspensions predicted lineup official",
                impact_area="lineup_availability",
                factor_label="Lineup Availability",
            ),
            EvidenceQuery(
                text=f"{base} starting XI confirmed lineup squad availability",
                impact_area="lineup_availability",
                factor_label="Lineup Availability",
            ),
            EvidenceQuery(
                text=f"{base} leaked lineup predicted XI rumors local media fan forum reddit",
                impact_area="lineup_availability",
                factor_label="Lineup Availability",
            ),
            EvidenceQuery(
                text=f"{base} training report unavailable doubtful late fitness test",
                impact_area="lineup_availability",
                factor_label="Lineup Availability",
            ),
            EvidenceQuery(
                text=f"{base} tactical preview key matchup",
                impact_area="tactical_matchup",
                factor_label="Tactical Matchup",
            ),
            EvidenceQuery(
                text=f"{base} tactical analysis fan forum local media style matchup pressing transition set pieces",
                impact_area="tactical_matchup",
                factor_label="Tactical Matchup",
            ),
            EvidenceQuery(
                text=f"{base} predicted tactics formation matchup strengths weaknesses",
                impact_area="tactical_matchup",
                factor_label="Tactical Matchup",
            ),
            EvidenceQuery(
                text=f"{base} referee appointment assigned referee official",
                impact_area="referee_environment",
                factor_label="Referee & Environment",
            ),
            EvidenceQuery(
                text=f"{base} venue weather kickoff conditions",
                impact_area="referee_environment",
                factor_label="Referee & Environment",
            ),
        ]

    def _query_base(self, request: PredictionRequest) -> str:
        competitors = request.context.get("competitors")
        if isinstance(competitors, list) and len(competitors) >= 2:
            return " ".join(str(item) for item in competitors[:2])
        return request.question
