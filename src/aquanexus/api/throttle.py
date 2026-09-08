"""A bounded cache and a token bucket for the explanation endpoint.

`/explain` costs roughly 190 ms of SHAP per call and nothing stopped a caller
issuing them in a loop. Two small, dependency-free pieces address that:

* :class:`BoundedCache` - explanations are a deterministic function of the
  feature row, so the same state asked twice can be answered from memory. A
  dashboard that re-explains the state a user is nudging back and forth gets
  most of its answers free.
* :class:`TokenBucket` - a ceiling on how fast the *expensive* path can be
  entered. Cache hits do not spend a token, because the limit exists to protect
  CPU and a hit costs none.

Both are per-process. With several uvicorn workers each has its own, so the
effective limit is the configured rate times the worker count - stated here
rather than discovered later. Anything stricter needs shared state (Redis, or a
gateway), which is a deployment decision rather than an application one.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any

from aquanexus.logger import get_logger

log = get_logger("api.throttle")


class BoundedCache:
    """A small thread-safe LRU.

    `functools.lru_cache` would do the same job, but this keeps hit and miss
    counts, which are the numbers that say whether caching was worth it, and can
    be cleared per model when a registry reloads.
    """

    def __init__(self, max_entries: int = 256):
        self.max_entries = max_entries
        self._entries: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self.hits += 1
                return self._entries[key]
            self.misses += 1
            return None

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = 0
            self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def stats(self) -> dict[str, int]:
        return {"entries": len(self._entries), "hits": self.hits,
                "misses": self.misses, "max_entries": self.max_entries}


class TokenBucket:
    """Per-key rate limiting, refilling continuously.

    A bucket rather than a fixed window: a caller who has been idle can spend a
    short burst, which is what an interactive dashboard actually does, while the
    sustained rate stays capped.
    """

    def __init__(self, rate_per_minute: int, burst: int | None = None,
                 clock=time.monotonic):
        self.rate_per_second = rate_per_minute / 60.0
        self.burst = burst if burst is not None else max(1, rate_per_minute // 3)
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def take(self, key: str) -> bool:
        """Spend one token for ``key``. False when the bucket is empty."""
        now = self._clock()
        with self._lock:
            tokens, last = self._buckets.get(key, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate_per_second)

            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False

            self._buckets[key] = (tokens - 1.0, now)
            return True

    def retry_after(self, key: str) -> int:
        """Whole seconds until one token is available, at least 1."""
        with self._lock:
            tokens, _ = self._buckets.get(key, (float(self.burst), self._clock()))
        if tokens >= 1.0 or self.rate_per_second <= 0:
            return 1
        return max(1, int((1.0 - tokens) / self.rate_per_second) + 1)

    def clear(self) -> None:
        with self._lock:
            self._buckets.clear()
