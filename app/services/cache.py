# app/services/cache.py
import functools
from typing import Any, Callable, Awaitable

def simple_cache(func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    cache = {}
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        key = (args, frozenset(kwargs.items()))
        if key in cache:
            return cache[key]
        result = await func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper
