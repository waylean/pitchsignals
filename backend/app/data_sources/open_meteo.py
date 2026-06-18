from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from app.data_sources.base import EvidenceQuery
from app.schemas import EvidenceItem, PredictionRequest


class OpenMeteoWeatherSource:
    """Public/free weather source for venue conditions.

    Requires request.context to include venue latitude and longitude. If the
    event time or venue is unavailable, it emits an explicit collection gap so
    the workflow can show the missing input without scoring it.
    """

    name = "open_meteo"
    base_url = "https://api.open-meteo.com/v1/forecast"

    async def collect(self, request: PredictionRequest, queries: list[EvidenceQuery]) -> list[EvidenceItem]:
        if request.domain != "football":
            return []

        latitude = self._number_context(request.context, "venue_latitude", "latitude", "lat")
        longitude = self._number_context(request.context, "venue_longitude", "longitude", "lon", "lng")
        if latitude is None or longitude is None:
            return [self._gap("Venue latitude/longitude missing; weather cannot be collected.")]
        if request.event_time is None:
            return [self._gap("Event time missing; kickoff weather cannot be collected.")]

        event_time = request.event_time
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=timezone.utc)
        date_text = event_time.date().isoformat()
        params: dict[str, Any] = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m",
            "timezone": "UTC",
            "start_date": date_text,
            "end_date": date_text,
        }

        try:
            async with httpx.AsyncClient(timeout=20, headers={"User-Agent": "ForecastIntelligence/0.1"}) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            return [self._gap(f"Open-Meteo collection failed: {exc}")]

        weather = self._nearest_hour(data, event_time)
        if not weather:
            return [self._gap("Open-Meteo returned no hourly weather near kickoff.")]

        claim = (
            f"Kickoff weather near venue: {weather['temperature']}C, "
            f"{weather['humidity']}% humidity, {weather['precipitation_probability']}% precipitation probability, "
            f"wind {weather['wind_speed']} km/h."
        )
        weather_risk = self._weather_risk(weather)
        return [
            EvidenceItem(
                claim=claim,
                source=self.name,
                source_url=str(response.url),
                source_query=f"Open-Meteo forecast {latitude},{longitude} {event_time.isoformat()}",
                evidence_stage="verified_candidate",
                raw_excerpt=claim,
                verifier_notes=["Parsed from public Open-Meteo forecast API."],
                impact_area="referee_environment",
                source_reliability=0.84,
                recency_score=0.88,
                corroboration_count=1,
                confidence=round(0.64 + weather_risk * 0.12, 4),
            )
        ]

    def _nearest_hour(self, data: dict[str, Any], event_time: datetime) -> dict[str, float] | None:
        hourly = data.get("hourly") or {}
        times = hourly.get("time") or []
        if not times:
            return None

        target = event_time.replace(minute=0, second=0, microsecond=0).isoformat(timespec="minutes")
        try:
            index = times.index(target)
        except ValueError:
            event_naive = event_time.replace(tzinfo=None)
            parsed = [datetime.fromisoformat(value) for value in times]
            index = min(range(len(parsed)), key=lambda idx: abs((parsed[idx] - event_naive).total_seconds()))

        def value(key: str, default: float = 0) -> float:
            values = hourly.get(key) or []
            if index >= len(values) or values[index] is None:
                return default
            return float(values[index])

        return {
            "temperature": value("temperature_2m"),
            "humidity": value("relative_humidity_2m"),
            "precipitation_probability": value("precipitation_probability"),
            "wind_speed": value("wind_speed_10m"),
        }

    def _weather_risk(self, weather: dict[str, float]) -> float:
        wind = min(weather["wind_speed"] / 45, 1)
        rain = min(weather["precipitation_probability"] / 100, 1)
        temperature = weather["temperature"]
        heat_or_cold = 0
        if temperature > 30:
            heat_or_cold = min((temperature - 30) / 12, 1)
        elif temperature < 2:
            heat_or_cold = min((2 - temperature) / 12, 1)
        return max(wind, rain, heat_or_cold)

    def _number_context(self, context: dict[str, Any], *keys: str) -> float | None:
        for key in keys:
            if key in context:
                try:
                    return float(context[key])
                except (TypeError, ValueError):
                    return None
        venue = context.get("venue")
        if isinstance(venue, dict):
            for key in keys:
                if key in venue:
                    try:
                        return float(venue[key])
                    except (TypeError, ValueError):
                        return None
        return None

    def _gap(self, message: str) -> EvidenceItem:
        return EvidenceItem(
            claim=message,
            source=self.name,
            source_url="https://open-meteo.com/",
            source_query="Open-Meteo venue weather",
            evidence_stage="collection_gap",
            impact_area="referee_environment",
            source_reliability=0.8,
            recency_score=0.0,
            confidence=0.0,
        )
