"""Small reusable, thread-safe rate limiter utilities."""

from __future__ import annotations

import threading
import time
from collections import deque


class SlidingWindowRateLimiter:
    """Block until a call slot is available within a fixed time window."""

    def __init__(self, *, max_calls: int, window_seconds: float) -> None:
        if max_calls <= 0:
            raise ValueError("max_calls must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._max_calls = max_calls
        self._window_seconds = window_seconds
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait_for_slot(self) -> None:
        """Wait until issuing one more call is within the configured rate window."""
        while True:
            sleep_for = 0.0
            now = time.monotonic()
            with self._lock:
                cutoff = now - self._window_seconds
                while self._calls and self._calls[0] <= cutoff:
                    self._calls.popleft()

                if len(self._calls) < self._max_calls:
                    self._calls.append(now)
                    return

                sleep_for = max(0.0, self._calls[0] + self._window_seconds - now)

            if sleep_for > 0:
                time.sleep(sleep_for)
