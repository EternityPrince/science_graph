import time
import asyncio

class AsyncRateLimiter:
    """Controls frequency of asynchronous operations to fit rate-limits (RPM)."""
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.last_call_time = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        """Suspends caller task if rate-limit interval hasn't been met yet."""
        if self.interval <= 0:
            return
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call_time
            if elapsed < self.interval:
                await asyncio.sleep(self.interval - elapsed)
            self.last_call_time = time.monotonic()
