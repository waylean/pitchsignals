# Open Source Checklist

Before publishing:

1. Choose a license.
2. Use placeholder values in `.env.example`.
3. Keep local runtime secrets in `backend/.env`.
4. Run backend compile check.
5. Run frontend build.
6. Test the app with valid API keys.
7. Test the app with missing `THE_ODDS_API_KEY`.
8. Test the app with `LLM_ENABLED=false`.
9. Make sure the README clearly says this is a research tool, not betting advice.
10. Add screenshots only if they do not expose private keys or account data.

Recommended first release label:

```text
0.1.0-research-preview
```
