from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    llm_provider: str = "openai_compatible"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4.1-mini"
    llm_enabled: bool = True
    llm_timeout_seconds: float = 120
    llm_max_tokens: int = 900

    the_odds_api_key: str | None = None
    the_odds_api_default_sport_key: str = "soccer_fifa_world_cup"
    the_odds_api_schedule_sport_keys: str = "soccer_fifa_world_cup,soccer_uefa_champs_league,soccer_epl,soccer_spain_la_liga,soccer_italy_serie_a,soccer_germany_bundesliga,soccer_france_ligue_one,soccer_usa_mls"
    the_odds_api_regions: str = "us,uk,eu"

    agent_reach_enabled: bool = True
    strict_workflow: bool = True
    prediction_ledger_path: str = "work/prediction_ledger.jsonl"


settings = Settings()
