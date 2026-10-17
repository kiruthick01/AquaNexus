"""Smoke-check a running AquaNexus API.

    python scripts/verify_deployment.py                       # localhost:8000
    python scripts/verify_deployment.py --url http://host:8000

Every check prints PASS, FAIL or SKIP with the reason. A degraded service - one that
started without models - fails the model checks and says so rather than reporting a
healthy API that cannot predict anything. Exit status is 0 only if every check that
ran passed.
"""

from __future__ import annotations

import argparse
import sys
import time

REQUEST = {
    "date": "2023-07-10",
    "mode": "geo",
    "location": {"latitude": 35.8, "longitude": 139.8},
    "weather": {"air_temp": 25.0},
}

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Checks:
    """Runs checks in order, remembering what failed."""

    def __init__(self, client, url: str):
        self.client = client
        self.url = url.rstrip("/")
        self.failures = 0
        self.skipped = 0

    def record(self, status: str, name: str, detail: str = "") -> None:
        print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))
        if status == FAIL:
            self.failures += 1
        elif status == SKIP:
            self.skipped += 1

    def get(self, path: str, **kwargs):
        return self.client.get(f"{self.url}{path}", timeout=30, **kwargs)

    def post(self, path: str, payload: dict):
        return self.client.post(f"{self.url}{path}", json=payload, timeout=60)


def check_reachable(checks: Checks) -> bool:
    try:
        started = time.perf_counter()
        response = checks.get("/health")
        elapsed = (time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001 - any transport failure is the answer
        checks.record(FAIL, "service reachable", f"{type(exc).__name__}: {exc}")
        return False
    if response.status_code != 200:
        checks.record(FAIL, "service reachable", f"/health returned {response.status_code}")
        return False
    checks.record(PASS, "service reachable", f"/health in {elapsed:.0f} ms")
    return True


def check_health(checks: Checks) -> bool:
    body = checks.get("/health").json()
    if body["status"] != "ok":
        checks.record(FAIL, "models loaded", f"status {body['status']}: {body.get('detail')}")
        return False
    checks.record(PASS, "models loaded", ", ".join(body["models_loaded"]))
    return True


def check_docs(checks: Checks) -> None:
    for path in ("/docs", "/openapi.json"):
        code = checks.get(path).status_code
        checks.record(PASS if code == 200 else FAIL, f"{path} served", f"HTTP {code}")


def check_models(checks: Checks) -> None:
    """Every model must report the held-out accuracy it is presented with."""
    response = checks.get("/models")
    if response.status_code != 200:
        return checks.record(FAIL, "models described", f"HTTP {response.status_code}")
    models = response.json()
    ok = bool(models) and all("held_out_strict" in m and m["held_out_note"] for m in models)
    checks.record(
        PASS if ok else FAIL,
        "models described with held-out accuracy",
        ", ".join(m["mode"] for m in models),
    )


def check_predict(checks: Checks) -> None:
    response = checks.post("/predict", REQUEST)
    if response.status_code != 200:
        return checks.record(FAIL, "predict", f"HTTP {response.status_code}")
    body = response.json()
    ok = 0.0 <= body["prediction"] <= 25.0 and body["lower"] < body["prediction"] < body["upper"]
    checks.record(
        PASS if ok else FAIL,
        "predict returns a value inside its interval",
        f"{body['prediction']:.2f} mg/L [{body['lower']:.2f}, {body['upper']:.2f}]",
    )


def check_explain(checks: Checks) -> None:
    response = checks.post("/predict", REQUEST | {"explain": True})
    if response.status_code != 200:
        return checks.record(FAIL, "explain", f"HTTP {response.status_code}")
    body = response.json()
    gap = abs(sum(body["contributions"].values()) - body["prediction"])
    checks.record(
        PASS if gap < 1e-4 else FAIL, "contributions sum to the prediction", f"difference {gap:.1e}"
    )


def check_batch(checks: Checks) -> None:
    response = checks.post("/batch_predict", {"requests": [REQUEST] * 5})
    ok = response.status_code == 200 and response.json()["n"] == 5
    checks.record(PASS if ok else FAIL, "batch predict", f"HTTP {response.status_code}")


def check_error_handling(checks: Checks) -> None:
    """A malformed request must come back as JSON, not an HTML error page."""
    response = checks.post("/predict", {"date": "2023-07-10"})
    is_json = response.headers.get("content-type", "").startswith("application/json")
    checks.record(
        PASS if response.status_code == 422 and is_json else FAIL,
        "request without a place rejected as JSON",
        f"HTTP {response.status_code}, {response.headers.get('content-type')}",
    )


def check_latency(checks: Checks, rounds: int = 20) -> None:
    checks.post("/predict", REQUEST)  # warm
    started = time.perf_counter()
    for _ in range(rounds):
        checks.post("/predict", REQUEST)
    per_call = (time.perf_counter() - started) * 1000 / rounds
    checks.record(PASS, "warm prediction latency", f"{per_call:.1f} ms per call")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the API")
    args = parser.parse_args()

    try:
        import httpx
    except ImportError:
        print("httpx is required: pip install '.[api]'")
        return 2

    print(f"AquaNexus deployment check -> {args.url}\n")
    with httpx.Client() as client:
        checks = Checks(client, args.url)
        if not check_reachable(checks):
            print("\nservice is not answering; nothing else can be checked")
            return 1
        healthy = check_health(checks)
        check_docs(checks)
        check_error_handling(checks)
        if healthy:
            check_models(checks)
            check_predict(checks)
            check_explain(checks)
            check_batch(checks)
            check_latency(checks)
        else:
            checks.record(
                SKIP,
                "model-dependent checks",
                "no models loaded; mount ./data or run scripts/save_do_model.py",
            )

    print(f"\n{checks.failures} failure(s), {checks.skipped} skipped")
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())
