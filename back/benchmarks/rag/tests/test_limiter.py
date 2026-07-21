import pytest
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


@pytest.mark.asyncio
async def test_async_rate_limiter_can_acquire_and_has_capacity():
    # Unlimited rate limiter (RPM=0)
    unlimited_limiter = AsyncRateLimiter(0)
    assert unlimited_limiter.can_acquire() is True
    assert unlimited_limiter.has_capacity() is True

    # 60 RPM = 1 call per second
    limiter = AsyncRateLimiter(60)
    assert limiter.can_acquire() is True
    assert limiter.has_capacity() is True

    # After wait call, last_call_time is updated to now, capacity should be False immediately
    await limiter.wait()
    assert limiter.can_acquire() is False
    assert limiter.has_capacity() is False

