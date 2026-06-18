from app.domain_packs.base import DomainPack, FactorDefinition


FOOTBALL_PACK = DomainPack(
    key="football",
    label="Football / Soccer",
    description="Match, tournament, scoreline, and qualification prediction.",
    outcome_types=["binary", "three_way", "scoreline", "multi_class"],
    default_sources=[
        "official_team_sites",
        "fifa_uefa_afc_conmebol",
        "odds_markets",
        "sports_data_apis",
        "trusted_journalists",
        "press_conferences",
        "social_discussion",
    ],
    known_failure_modes=[
        "Late lineup changes dominate stale analysis.",
        "Unofficial team news can be noisy and must be aggregated, not trusted directly.",
        "Odds already price in public information.",
        "Scoreline predictions are much less stable than outcome predictions.",
    ],
    factors=[
        FactorDefinition(
            key="market_odds",
            label="Market Odds",
            description="Opening, current, and closing odds with implied probability and movement.",
            default_weight=0.28,
            half_life_hours=8,
            evidence_queries=["bookmaker odds", "closing line", "odds movement"],
        ),
        FactorDefinition(
            key="team_strength",
            label="Team Strength",
            description="Elo, ranking, xG profile, attacking and defensive quality.",
            default_weight=0.22,
            half_life_hours=240,
            evidence_queries=["Elo rating", "xG", "recent match statistics"],
        ),
        FactorDefinition(
            key="lineup_availability",
            label="Lineup Availability",
            description="Injuries, suspensions, likely starting XI, fatigue, travel load.",
            default_weight=0.20,
            half_life_hours=4,
            evidence_queries=["injury report", "training absence", "predicted lineup", "leaked lineup", "team news rumors"],
        ),
        FactorDefinition(
            key="tactical_matchup",
            label="Tactical Matchup",
            description="Style interaction, pressing, transition defense, set pieces, substitutions.",
            default_weight=0.14,
            half_life_hours=72,
            evidence_queries=["tactical preview", "coach press conference", "fan tactical analysis", "local media tactical notes"],
        ),
        FactorDefinition(
            key="referee_environment",
            label="Referee & Environment",
            description="Referee tendency, venue, weather, pitch, crowd, time zone.",
            default_weight=0.08,
            half_life_hours=12,
            evidence_queries=["referee appointment", "weather", "venue conditions"],
        ),
        FactorDefinition(
            key="sentiment_narrative",
            label="Sentiment & Narrative",
            description="Media pressure, fan expectations, social signal shifts, confidence tone.",
            default_weight=0.08,
            half_life_hours=6,
            evidence_queries=["fan sentiment", "media narrative", "team morale"],
        ),
    ],
)
