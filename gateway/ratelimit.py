"""Per-key token-bucket rate limiter (in-memory; one gateway replica = one bucket set).

For multi-replica deployments put the bucket in Redis; the interface stays the same.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class TokenBucketLimiter:
    def __init__(self, clock=time.monotonic):
        self._buckets: dict[int, _Bucket] = {}
        self._lock = threading.Lock()
        self._clock = clock

    def allow(self, key_id: int, per_minute: int) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds). Capacity = per_minute, refill = per_minute/60 per s."""
        if per_minute <= 0:
            return False, 60.0
        now = self._clock()
        rate = per_minute / 60.0
        with self._lock:
            b = self._buckets.get(key_id)
            if b is None:
                b = _Bucket(tokens=float(per_minute), updated=now)
                self._buckets[key_id] = b
            else:
                b.tokens = min(float(per_minute), b.tokens + (now - b.updated) * rate)
                b.updated = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, 0.0
            retry = (1.0 - b.tokens) / rate
            return False, math.ceil(retry * 100) / 100

    def reset(self, key_id: int | None = None) -> None:
        with self._lock:
            if key_id is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key_id, None)
