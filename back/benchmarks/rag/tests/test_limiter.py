import pytest
import asyncio
import time
from core.limiter import AsyncRateLimiter

@pytest.mark.asyncio
async def test_async_rate_limiter_disabled():
    limiter = AsyncRateLimiter(0)
    t0 = time.monotonic()
    await limiter.wait()
    assert time.monotonic() - t0 < 0.1

@pytest.mark.asyncio
async def test_async_rate_limiter_waits():
    limiter = AsyncRateLimiter(60) # 1 call per second
    assert limiter.interval == 1.0
    
    # First call sets last_call_time
    await limiter.wait()
    
    t0 = time.monotonic()
    # Second call must wait for the interval
    await limiter.wait()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.8
