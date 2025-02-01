# app/utils/helpers.py
import functools
import asyncio
import random
import logging
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

def async_retry(retries: int = 3, backoff_in_seconds: float = 1.0, max_backoff: float = 10.0) -> Callable:
    def decorator(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            delay = backoff_in_seconds
            last_exception = None
            for attempt in range(1, retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt}/{retries} failed: {e}")
                    last_exception = e
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, max_backoff) + random.uniform(0, 0.1)
            raise last_exception
        return wrapper
    return decorator
