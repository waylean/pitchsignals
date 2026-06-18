# Contributing

Contributions should keep the project simple.

Good contributions:

1. Add a public/free data source.
2. Improve source reliability scoring.
3. Add reproducible backtests.
4. Improve degraded mode when a data source is missing.
5. Make the UI clearer for non-technical users.

Avoid:

1. Hardcoding private API keys.
2. Adding paid-only data as a required dependency.
3. Making the LLM directly output final probabilities.
4. Adding complex UI before the data layer is reliable.
5. Claiming betting profitability without evidence.

## Development Checks

Backend:

```bash
cd backend
python -m compileall app
```

Frontend:

```bash
cd frontend
npm run build
```

