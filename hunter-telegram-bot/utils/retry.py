"""Hunter Bot — Retry Decorator"""
import asyncio
import functools
import logging
from typing import Callable, Type, Tuple

logger = logging.getLogger("hunter")


def async_retry(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    timeout: float = 10.0,
):
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs), timeout=timeout
                    )
                except exceptions as e:
                    last_exception = e
                    logger.warning(f"[RETRY {attempt}/{max_retries}] {func.__qualname__}: {e}")
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            raise last_exception
        return wrapper
    return decorator
