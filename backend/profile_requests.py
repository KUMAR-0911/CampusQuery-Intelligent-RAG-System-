"""Endpoint profiling utility to track execution duration and identify hotspots."""
import time
import functools
import inspect
import logging

logger = logging.getLogger("campusquery.profiler")

def profile_endpoint(func):
    """Decorator to measure and log endpoint execution duration."""
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                if duration_ms > 500.0:
                    print(f"[profiler:warn] {func.__name__} took {duration_ms:.2f}ms (high latency)")
                else:
                    logger.debug(f"[profiler] {func.__name__} took {duration_ms:.2f}ms")
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.perf_counter() - start) * 1000.0
                if duration_ms > 500.0:
                    print(f"[profiler:warn] {func.__name__} took {duration_ms:.2f}ms (high latency)")
                else:
                    logger.debug(f"[profiler] {func.__name__} took {duration_ms:.2f}ms")
        return sync_wrapper
