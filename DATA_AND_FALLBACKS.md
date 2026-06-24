# Data Sources and Fallbacks

The project should prefer public or free data sources.

## Required For Best Experience

### The Odds API

Used for:

1. Upcoming fixture discovery
2. 1X2 market odds
3. Bookmaker consensus
4. Market-implied probabilities

Free tier limitations:

1. Quota limits
2. Sport availability changes
3. Some events may have no 1X2 market
4. Some regions may return fewer bookmakers

Fallback behavior:

1. If no API key exists, schedule fetching returns a missing-key status.
2. If odds are unavailable for a match, market odds should be marked missing.
3. Confidence should decrease.
4. Other factors may still contribute if the match is supplied through another route.

## LLM API

Used for:

1. Structuring unofficial or difficult text
2. Extracting low-confidence candidate features
3. Reviewing evidence coverage

The LLM should not directly decide final probabilities.

Fallback behavior:

1. If `LLM_ENABLED=false`, skip LLM structuring.
2. If `LLM_API_KEY` is missing, return skipped status.
3. Prediction can still run, but difficult information becomes less useful.
4. Confidence should decrease when key information cannot be structured.

## Public Football Sources

Currently useful:

1. World Football Elo
2. Football-Data.co.uk historical CSVs
3. Public previews and match reports
4. Public weather APIs where venue/time are known
5. StatsBomb Open Data for historical replay research
6. Optional Zhihu Open Platform search for Chinese-language context

## Optional Zhihu Search

Zhihu Open Platform can be used as an optional search enhancement.

Docs: [https://developer.zhihu.com/docs?key=authorization](https://developer.zhihu.com/docs?key=authorization)

Used for:

1. Chinese-language football discussion
2. Match background and context
3. Public community signals around players, teams, tactics, and narratives

Configuration:

```env
ZHIHU_SEARCH_ENABLED=true
ZHIHU_ACCESS_SECRET=your-zhihu-access-secret
ZHIHU_SEARCH_MODE=zhihu
```

Fallback behavior:

1. If the key is missing, Zhihu search is skipped.
2. If authorization fails, the error is recorded as collection metadata.
3. The default public search path remains available.
4. Zhihu results are treated as candidate evidence, not verified facts.

## Difficult Data

These are valuable but hard to structure:

1. Confirmed lineups
2. Predicted lineups
3. Injury rumors
4. Referee appointment
5. Tactical matchup
6. Motivation, rotation, and media pressure

Recommended treatment:

1. Use source timestamps.
2. Use source reliability scores.
3. Cap confidence for unofficial claims.
4. Do not let rumor dominate the model.
5. Show lower prediction confidence when these signals are missing.
