from app.domain_packs.base import DomainPack, FactorDefinition


BASKETBALL_PACK = DomainPack(
    key="basketball",
    label="Basketball",
    description="Basketball game, spread, total, player prop, and series prediction.",
    outcome_types=["binary", "multi_class", "numeric"],
    default_sources=[
        "official_injury_reports",
        "odds_markets",
        "team_beat_reporters",
        "advanced_box_scores",
        "lineup_trackers",
        "social_discussion",
    ],
    known_failure_modes=[
        "Late injury and rest news can invalidate earlier estimates.",
        "Back-to-back fatigue and travel are often underpriced in generic models.",
        "Player prop predictions need minute projections, not just player quality.",
    ],
    factors=[
        FactorDefinition(
            key="market_odds",
            label="Market Odds",
            description="Moneyline, spread, total, opening and current odds movement.",
            default_weight=0.28,
            half_life_hours=4,
            evidence_queries=["moneyline spread total odds movement"],
        ),
        FactorDefinition(
            key="team_strength",
            label="Team Strength",
            description="Net rating, offensive rating, defensive rating, pace, shot profile.",
            default_weight=0.22,
            half_life_hours=120,
            evidence_queries=["net rating offensive defensive rating pace"],
        ),
        FactorDefinition(
            key="availability_minutes",
            label="Availability & Minutes",
            description="Injury status, rest, rotation, likely minutes, load management.",
            default_weight=0.18,
            half_life_hours=3,
            evidence_queries=["injury report probable questionable minutes restriction"],
        ),
        FactorDefinition(
            key="matchup_style",
            label="Matchup Style",
            description="Switching, rim pressure, three-point volume, rebounding, transition.",
            default_weight=0.12,
            half_life_hours=48,
            evidence_queries=["matchup preview defensive scheme rebounding"],
        ),
        FactorDefinition(
            key="schedule_fatigue",
            label="Schedule & Fatigue",
            description="Back-to-back, travel, altitude, home stand, rest advantage.",
            default_weight=0.10,
            half_life_hours=24,
            evidence_queries=["back to back travel rest advantage"],
        ),
        FactorDefinition(
            key="chemistry_role_fit",
            label="Chemistry & Role Fit",
            description="Lineup continuity, usage changes, coach trust, locker-room signals.",
            default_weight=0.06,
            half_life_hours=120,
            evidence_queries=["lineup continuity usage rate coach trust chemistry"],
        ),
        FactorDefinition(
            key="sentiment_narrative",
            label="Sentiment & Narrative",
            description="Local reporting, morale, pressure, trade rumors, public overreaction.",
            default_weight=0.04,
            half_life_hours=8,
            evidence_queries=["locker room morale trade rumors local report"],
        ),
    ],
)

