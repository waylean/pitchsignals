from app.domain_packs.base import DomainPack, FactorDefinition


ENTERTAINMENT_PACK = DomainPack(
    key="entertainment",
    label="Entertainment / Box Office",
    description="Movie box office, awards, album performance, and cultural event prediction.",
    outcome_types=["binary", "multi_class", "numeric"],
    default_sources=["ticketing", "reviews", "social_platforms", "search_trends", "distribution_data"],
    known_failure_modes=[
        "Social buzz can be loud but narrow.",
        "Review embargo timing distorts early sentiment.",
        "Distribution scale can dominate audience intent.",
    ],
    factors=[
        FactorDefinition(key="pre_sales", label="Pre-sales", description="Advance ticket sales and pace.", default_weight=0.30),
        FactorDefinition(key="distribution", label="Distribution", description="Screens, showtimes, geography, release window.", default_weight=0.20),
        FactorDefinition(key="audience_intent", label="Audience Intent", description="Search, trailer engagement, social conversion.", default_weight=0.18),
        FactorDefinition(key="critical_reception", label="Critical Reception", description="Reviews and audience scores.", default_weight=0.12),
        FactorDefinition(key="competition", label="Competition", description="Competing releases and calendar effects.", default_weight=0.12),
        FactorDefinition(key="franchise_context", label="Franchise Context", description="Brand strength, cast draw, fatigue.", default_weight=0.08),
    ],
)

