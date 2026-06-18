from app.domain_packs.base import DomainPack, FactorDefinition


ELECTIONS_PACK = DomainPack(
    key="elections",
    label="Elections / Polling",
    description="Election winner, referendum, turnout, and polling movement prediction.",
    outcome_types=["binary", "multi_class", "numeric"],
    default_sources=["polling_aggregators", "official_results", "campaign_news", "demographics", "prediction_markets"],
    known_failure_modes=[
        "Poll quality varies sharply by methodology.",
        "Turnout assumptions can dominate polling average.",
        "Late swings may not be captured in low-frequency polling.",
    ],
    factors=[
        FactorDefinition(key="polling_average", label="Polling Average", description="Weighted, adjusted polling trend.", default_weight=0.35),
        FactorDefinition(key="fundamentals", label="Fundamentals", description="Economy, incumbency, approval, demographics.", default_weight=0.20),
        FactorDefinition(key="turnout_model", label="Turnout Model", description="Likely voter composition and enthusiasm.", default_weight=0.16),
        FactorDefinition(key="campaign_events", label="Campaign Events", description="Debates, scandals, endorsements, crises.", default_weight=0.12),
        FactorDefinition(key="market_signal", label="Market Signal", description="Prediction market and betting odds.", default_weight=0.10),
        FactorDefinition(key="ground_operation", label="Ground Operation", description="Field organization and mobilization.", default_weight=0.07),
    ],
)

