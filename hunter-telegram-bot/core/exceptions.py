"""Hunter Bot — Custom Exceptions"""


class HunterError(Exception):
    """Base exception."""
    pass


class ProviderError(HunterError):
    """External API/provider failure."""
    def __init__(self, message: str, provider: str = "unknown", retryable: bool = True):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class DataInsufficientError(HunterError):
    """Required data missing or unreliable."""
    pass


class SessionError(HunterError):
    """Market session boundary issue."""
    pass


class NewsValidationError(HunterError):
    """News failed quality/materiality gates."""
    pass
