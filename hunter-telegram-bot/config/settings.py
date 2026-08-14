"""Central configuration for Hunter Bot."""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)
    telegram_bot_token: str = Field(default="test", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="test", alias="TELEGRAM_CHAT_ID")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    polygon_api_key: Optional[str] = Field(default=None, alias="POLYGON_API_KEY")
    finnhub_api_key: Optional[str] = Field(default=None, alias="FINNHUB_API_KEY")
    hunter_min_score: int = Field(default=70, ge=0, le=100, alias="HUNTER_MIN_SCORE")
    hunter_min_data_confidence: int = Field(default=60, ge=0, le=100, alias="HUNTER_MIN_DATA_CONFIDENCE")
    max_watchlist_price: float = Field(default=100.0, alias="MAX_WATCHLIST_PRICE")
    account_size: Optional[float] = Field(default=None, alias="ACCOUNT_SIZE")
    risk_per_trade_pct: float = Field(default=0.5, ge=0.05, le=5.0, alias="RISK_PER_TRADE_PCT")
    options_enabled: bool = Field(default=True, alias="OPTIONS_ENABLED")
    market_timezone: str = Field(default="America/New_York", alias="MARKET_TIMEZONE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    memory_db_path: str = Field(default="data/hunter_memory.sqlite3", alias="MEMORY_DB_PATH")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed: raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    @property
    def has_polygon(self): return bool(self.polygon_api_key and self.polygon_api_key.strip())
    @property
    def has_finnhub(self): return bool(self.finnhub_api_key and self.finnhub_api_key.strip())

SETTINGS = Settings()
