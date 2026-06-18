from app.domain_packs.base import DomainPack, FactorDefinition


FINANCE_PACK = DomainPack(
    key="finance",
    label="Finance / Markets",
    description="Asset direction, earnings reaction, macro event, and risk prediction.",
    outcome_types=["binary", "multi_class", "numeric"],
    default_sources=["market_data", "filings", "earnings_calls", "macro_data", "news", "social_discussion"],
    known_failure_modes=[
        "Market prices often incorporate obvious public information.",
        "Narratives after price moves can be misleading.",
        "Regime shifts break historical correlations.",
    ],
    factors=[
        FactorDefinition(key="price_action", label="Price Action", description="Trend, volatility, liquidity.", default_weight=0.25),
        FactorDefinition(key="fundamentals", label="Fundamentals", description="Revenue, margins, balance sheet, guidance.", default_weight=0.22),
        FactorDefinition(key="macro_context", label="Macro Context", description="Rates, inflation, FX, policy, liquidity.", default_weight=0.18),
        FactorDefinition(key="positioning", label="Positioning", description="Options, short interest, flows, crowdedness.", default_weight=0.15),
        FactorDefinition(key="news_catalysts", label="News Catalysts", description="Earnings, regulation, product, litigation.", default_weight=0.12),
        FactorDefinition(key="sentiment", label="Sentiment", description="Analyst, media, and social tone.", default_weight=0.08),
    ],
)

