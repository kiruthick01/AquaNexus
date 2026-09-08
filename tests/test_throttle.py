"""Cache and rate-limit behaviour for the explanation endpoint.

The units are tested against a fake clock rather than by sleeping: a test that
waits for a token bucket to refill is a test nobody runs twice.
"""

from __future__ import annotations

import pytest

from aquanexus.api.throttle import BoundedCache, TokenBucket


class FakeClock:
    """A clock the test drives, so refill can be observed without waiting."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_returns_what_was_stored():
    cache = BoundedCache()
    assert cache.get("a") is None
    cache.put("a", 1)
    assert cache.get("a") == 1
    assert cache.stats == {"entries": 1, "hits": 1, "misses": 1, "max_entries": 256}


def test_cache_evicts_the_least_recently_used():
    """Bounded means bounded: an unbounded cache on request data is a leak."""
    cache = BoundedCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # "a" is now the most recent
    cache.put("c", 3)

    assert len(cache) == 2
    assert cache.get("a") == 1
    assert cache.get("b") is None, "the least recently used entry survived"


def test_cache_can_be_cleared():
    cache = BoundedCache()
    cache.put("a", 1)
    cache.clear()
    assert len(cache) == 0
    assert cache.stats["hits"] == 0


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def test_bucket_allows_a_burst_then_stops():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_minute=60, burst=3, clock=clock)

    assert [bucket.take("client") for _ in range(3)] == [True, True, True]
    assert bucket.take("client") is False


def test_bucket_refills_over_time():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_minute=60, burst=2, clock=clock)  # one per second

    bucket.take("client")
    bucket.take("client")
    assert bucket.take("client") is False

    clock.advance(1.0)
    assert bucket.take("client") is True, "a token should have refilled"


def test_bucket_does_not_refill_past_its_burst():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_minute=600, burst=2, clock=clock)

    clock.advance(3600)  # idle for an hour
    assert [bucket.take("c") for _ in range(3)] == [True, True, False]


def test_clients_are_limited_separately():
    """One noisy caller must not exhaust everyone else's budget."""
    clock = FakeClock()
    bucket = TokenBucket(rate_per_minute=60, burst=1, clock=clock)

    assert bucket.take("noisy") is True
    assert bucket.take("noisy") is False
    assert bucket.take("quiet") is True


def test_retry_after_is_a_usable_number_of_seconds():
    clock = FakeClock()
    bucket = TokenBucket(rate_per_minute=60, burst=1, clock=clock)

    bucket.take("client")
    wait = bucket.retry_after("client")
    assert wait >= 1

    clock.advance(wait)
    assert bucket.take("client") is True, "Retry-After promised too little time"


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aquanexus.api.app import app  # noqa: E402
from aquanexus.api.routes import explanations  # noqa: E402
from aquanexus.config import settings  # noqa: E402

models_present = pytest.mark.skipif(
    not (settings.MODELS_DIR / "manifest.json").is_file(),
    reason="trained models absent; run scripts/train_models.py",
)

STATE = {"water_temp": 21.0, "discharge": 8.0, "depth": 2.4, "velocity": 0.36,
         "month": 5}


@pytest.fixture
def fresh_throttle():
    """Each test starts with an empty cache and a full bucket."""
    explanations.explanation_cache.clear()
    explanations.explain_limiter.clear()
    yield
    explanations.explanation_cache.clear()
    explanations.explain_limiter.clear()


@models_present
def test_a_repeated_state_is_served_from_cache(fresh_throttle):
    payload = {"target": "dissolved_oxygen", "state": STATE}
    with TestClient(app) as client:
        first = client.post("/explain", json=payload).json()
        second = client.post("/explain", json=payload).json()

    assert first == second
    assert explanations.explanation_cache.stats["hits"] == 1


@models_present
def test_cache_hits_do_not_spend_the_rate_limit(fresh_throttle, monkeypatch):
    """The limit protects CPU, and an answer already computed costs none."""
    monkeypatch.setattr(explanations, "explain_limiter",
                        TokenBucket(rate_per_minute=60, burst=1))
    payload = {"target": "dissolved_oxygen", "state": STATE}

    with TestClient(app) as client:
        assert client.post("/explain", json=payload).status_code == 200  # spends it
        for _ in range(5):
            assert client.post("/explain", json=payload).status_code == 200


@models_present
def test_an_uncached_flood_is_refused_with_retry_after(fresh_throttle, monkeypatch):
    monkeypatch.setattr(explanations, "explain_limiter",
                        TokenBucket(rate_per_minute=60, burst=1))

    with TestClient(app) as client:
        first = client.post("/explain", json={"target": "dissolved_oxygen",
                                              "state": STATE})
        assert first.status_code == 200

        # A different state each time, so the cache cannot absorb them.
        refused = None
        for offset in range(1, 5):
            response = client.post("/explain", json={
                "target": "dissolved_oxygen",
                "state": {**STATE, "water_temp": 21.0 + offset},
            })
            if response.status_code == 429:
                refused = response
                break

    assert refused is not None, "the rate limit never engaged"
    assert int(refused.headers["Retry-After"]) >= 1
    assert "cache" in refused.json()["detail"], "the 429 should say how to avoid it"


@models_present
def test_rounding_noise_still_hits_the_same_entry(fresh_throttle):
    """A slider emitting 24.500000000000004 must not miss the cache."""
    with TestClient(app) as client:
        client.post("/explain", json={"target": "dissolved_oxygen",
                                      "state": {**STATE, "water_temp": 21.0}})
        client.post("/explain", json={
            "target": "dissolved_oxygen",
            "state": {**STATE, "water_temp": 21.0 + 4e-15},
        })

    assert explanations.explanation_cache.stats["hits"] == 1
