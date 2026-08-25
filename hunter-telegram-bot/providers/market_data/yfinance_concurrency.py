"""Hunter Bot — yfinance Concurrency Control

Provides a shared semaphore to limit concurrent yfinance calls across all providers,
preventing thread pool exhaustion.
"""
import asyncio
from typing import Optional

# Lazy semaphore creation to avoid event loop binding issues
_yfinance_semaphore: Optional[asyncio.Semaphore] = None

def get_yfinance_semaphore() -> asyncio.Semaphore:
    """Get or create the yfinance semaphore (lazy initialization)."""
    global _yfinance_semaphore
    if _yfinance_semaphore is None:
        _yfinance_semaphore = asyncio.Semaphore(4)
    return _yfinance_semaphore