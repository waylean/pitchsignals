from __future__ import annotations

from datetime import datetime
from typing import Any


FOOTBALL_HORIZON_PROFILES: dict[str, dict[str, Any]] = {
    "T-48h": {
        "max_hours_to_kickoff": 48,
        "weight_profile": {
            "market_odds": 0.36,
            "team_strength": 0.36,
            "lineup_availability": 0.10,
            "tactical_matchup": 0.09,
            "referee_environment": 0.07,
            "sentiment_narrative": 0.02,
        },
        "allowed_feature_types": [
            "odds",
            "elo_rating",
            "injury",
            "suspension",
            "predicted_lineup",
            "weather",
            "referee",
            "tactical_style",
            "sentiment",
        ],
    },
    "T-24h": {
        "max_hours_to_kickoff": 24,
        "weight_profile": {
            "market_odds": 0.40,
            "team_strength": 0.30,
            "lineup_availability": 0.14,
            "tactical_matchup": 0.08,
            "referee_environment": 0.06,
            "sentiment_narrative": 0.02,
        },
        "allowed_feature_types": [
            "odds",
            "elo_rating",
            "injury",
            "suspension",
            "predicted_lineup",
            "weather",
            "referee",
            "tactical_style",
            "sentiment",
        ],
    },
    "T-2h": {
        "max_hours_to_kickoff": 2,
        "weight_profile": {
            "market_odds": 0.46,
            "team_strength": 0.22,
            "lineup_availability": 0.18,
            "tactical_matchup": 0.08,
            "referee_environment": 0.04,
            "sentiment_narrative": 0.02,
        },
        "allowed_feature_types": [
            "odds",
            "elo_rating",
            "injury",
            "suspension",
            "predicted_lineup",
            "confirmed_lineup",
            "weather",
            "referee",
            "tactical_style",
            "sentiment",
        ],
    },
    "T-1h": {
        "max_hours_to_kickoff": 1,
        "weight_profile": {
            "market_odds": 0.50,
            "team_strength": 0.18,
            "lineup_availability": 0.20,
            "tactical_matchup": 0.06,
            "referee_environment": 0.04,
            "sentiment_narrative": 0.02,
        },
        "allowed_feature_types": [
            "odds",
            "elo_rating",
            "injury",
            "suspension",
            "predicted_lineup",
            "confirmed_lineup",
            "weather",
            "referee",
            "tactical_style",
            "sentiment",
        ],
    },
}


def apply_football_horizon_profile(context: dict[str, Any], event_time: datetime | None, deadline: datetime | None) -> str | None:
    requested = context.get("horizon_profile")
    if isinstance(requested, str) and requested in FOOTBALL_HORIZON_PROFILES:
        profile_id = requested
    else:
        profile_id = infer_horizon_profile(event_time, deadline)
    if not profile_id:
        return None

    profile = FOOTBALL_HORIZON_PROFILES[profile_id]
    context.setdefault("weight_profile", profile["weight_profile"])
    context["horizon_profile"] = profile_id
    context["allowed_feature_types"] = profile["allowed_feature_types"]
    return profile_id


def infer_horizon_profile(event_time: datetime | None, deadline: datetime | None) -> str | None:
    if not event_time or not deadline:
        return None
    hours = max((event_time - deadline).total_seconds() / 3600, 0)
    if hours <= 1:
        return "T-1h"
    if hours <= 2:
        return "T-2h"
    if hours <= 24:
        return "T-24h"
    return "T-48h"
