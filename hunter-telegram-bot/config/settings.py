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
    scan_interval_regular: int = Field(default=300, ge=30, le=3600, alias="SCAN_INTERVAL_REGULAR")
    scan_interval_extended: int = Field(default=900, ge=30, le=14400, alias="SCAN_INTERVAL_EXTENDED")
    scan_interval_closed: int = Field(default=1800, ge=30, le=86400, alias="SCAN_INTERVAL_CLOSED")
    telegram_commands_enabled: bool = Field(default=True, alias="TELEGRAM_COMMANDS_ENABLED")
    telegram_authorized_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_AUTHORIZED_CHAT_ID")
    discovery_enabled: bool = Field(default=True, alias="DISCOVERY_ENABLED")
    discovery_pool_size: int = Field(default=10, ge=1, le=50, alias="DISCOVERY_POOL_SIZE")
    discovery_cache_ttl: int = Field(default=180, ge=30, le=3600, alias="DISCOVERY_CACHE_TTL")
    discovery_min_abs_change: float = Field(default=3.0, ge=0.5, le=50.0, alias="DISCOVERY_MIN_ABS_CHANGE")
    discovery_max_candidates_per_source: int = Field(default=25, ge=1, le=100, alias="DISCOVERY_MAX_CANDIDATES_PER_SOURCE")
    # Stage 2 — realtime (safe defaults: disabled unless explicitly enabled)
    realtime_enabled: bool = Field(default=False, alias="REALTIME_ENABLED")
    realtime_max_age_seconds: int = Field(default=30, ge=5, le=300, alias="REALTIME_MAX_AGE_SECONDS")
    polygon_ws_enabled: bool = Field(default=False, alias="POLYGON_WS_ENABLED")

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

    @property
    def authorized_chat_ids(self) -> set:
        """Chat IDs allowed to issue commands. Defaults to the alert chat."""
        ids = {self.telegram_chat_id.strip()}
        extra = (self.telegram_authorized_chat_id or "").strip()
        if extra:
            ids.update(x.strip() for x in extra.split(",") if x.strip())
        return {i for i in ids if i}

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
