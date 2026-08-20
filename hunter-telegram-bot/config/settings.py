"""Central configuration for Hunter Bot."""
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

from core.exceptions import ConfigurationError


# Sentinel values that indicate a credential was never configured.
# These must be rejected at production startup.
PRODUCTION_INVALID_TELEGRAM = {"test", "changeme", "your-token", "your_telegram_bot_token_here", "your_telegram_chat_id_here", ""}
PRODUCTION_INVALID_POLYGON = {"test", "changeme", "your-api-key", ""}
PRODUCTION_INVALID_FINNHUB = {"test", "changeme", "your-api-key", ""}


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

    @field_validator("account_size", mode="before")
    @classmethod
    def _empty_account_size_to_none(cls, v):
        # An empty string (e.g. ACCOUNT_SIZE= in .env) must map to None,
        # not fail float parsing.
        if v is None:
            return None
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    @property
    def has_polygon(self) -> bool:
        key = self.polygon_api_key
        return bool(key and key.strip() and key.strip() not in PRODUCTION_INVALID_POLYGON)

    @property
    def has_finnhub(self) -> bool:
        key = self.finnhub_api_key
        return bool(key and key.strip() and key.strip() not in PRODUCTION_INVALID_FINNHUB)

    @property
    def telegram_configured(self) -> bool:
        """True when Telegram credentials are present and not placeholder values."""
        return (
            bool(self.telegram_bot_token and self.telegram_bot_token.strip())
            and self.telegram_bot_token.strip() not in PRODUCTION_INVALID_TELEGRAM
            and bool(self.telegram_chat_id and self.telegram_chat_id.strip())
            and self.telegram_chat_id.strip() not in PRODUCTION_INVALID_TELEGRAM
        )

    def validate_production(self) -> None:
        """Validate required configuration for production startup.

        Raises ConfigurationError with a clear message listing all problems.
        Never exposes secret values in the error message.
        """
        errors: List[str] = []

        # Telegram is always required — bot cannot function without it
        if not self.telegram_configured:
            missing = []
            if not self.telegram_bot_token or self.telegram_bot_token.strip() in PRODUCTION_INVALID_TELEGRAM:
                missing.append("TELEGRAM_BOT_TOKEN")
            if not self.telegram_chat_id or self.telegram_chat_id.strip() in PRODUCTION_INVALID_TELEGRAM:
                missing.append("TELEGRAM_CHAT_ID")
            if missing:
                errors.append(f"Missing or placeholder value for: {', '.join(missing)}")

        if errors:
            raise ConfigurationError(
                f"Configuration errors at startup:\n  - " + "\n  - ".join(errors)
            )


SETTINGS = Settings()
