# Roadmap

Keep the project simple.

## Version 0.1

Status: mostly done.

1. Local frontend
2. Backend prediction API
3. Today/tomorrow football schedule
4. The Odds API integration
5. LLM-assisted text structuring
6. Chinese/English frontend
7. Cached predictions when switching matches

## Version 0.2

Focus: make the data layer more dependable.

1. Add manual fixture input when The Odds API schedule is unavailable.
2. Add CSV fixture import.
3. Store odds snapshots locally.
4. Expose backend evidence detail through an admin/debug endpoint.
5. Add clearer missing-data warnings.

## Version 0.3

Focus: research quality.

1. Backtest more leagues.
2. Save pre-match snapshots with `available_at`.
3. Compare model probabilities with closing odds.
4. Track calibration by confidence bucket.
5. Add model leaderboard reports.

## Version 1.0

Release only when:

1. Setup is one-command or close to it.
2. The app works when odds are unavailable, with a clear degraded mode.
3. Historical backtests are reproducible.
4. Evidence provenance is queryable.
5. The frontend clearly distinguishes probability from confidence.

