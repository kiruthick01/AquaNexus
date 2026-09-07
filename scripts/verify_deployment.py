"""Smoke-check a running AquaNexus API.

Written for the container check that could not be run on the development
machine: `docker compose up`, then point this at the published port. It works
equally against a local `uvicorn` process, which is how it was tested.

    python scripts/verify_deployment.py                       # localhost:8000
    python scripts/verify_deployment.py --url http://host:8000

Every check prints PASS, FAIL or SKIP with the reason. A degraded service - one
that started without models - fails the model checks and says so rather than
reporting a healthy API that cannot predict anything.

Exit status is 0 only if every check that ran passed.
"""

from __future__ import annotations

import argparse
import sys
import time

STATE = {
    "water_temp": 22.0, "dissolved_oxygen": 6.8, "discharge": 9.5,
    "depth": 1.6, "velocity": 0.4, "month": 6,
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
        symbol = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}[status]
        print(f"  [{symbol}] {name}" + (f" - {detail}" if detail else ""))
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
        checks.record(FAIL, "models loaded",
                      f"status {body['status']}: {body.get('detail')}")
        return False
    checks.record(PASS, "models loaded", ", ".join(body["models_loaded"]))
    return True


def check_docs(checks: Checks) -> None:
    for path in ("/docs", "/openapi.json"):
        code = checks.get(path).status_code
        checks.record(PASS if code == 200 else FAIL, f"{path} served", f"HTTP {code}")


def check_provenance(checks: Checks) -> None:
    """The synthetic model must announce itself. This is the honesty contract."""
    models = {m["target"]: m for m in checks.get("/models").json()}
    if "hsi" not in models:
        return checks.record(SKIP, "synthetic labels declared", "hsi not served")

    labels = models["hsi"]["labels"]
    caveats = " ".join(models["hsi"]["caveats"]).upper()
    ok = labels == "SYNTHETIC" and "SYNTHETIC" in caveats
    checks.record(PASS if ok else FAIL, "synthetic labels declared",
                  f"labels={labels}")


def check_predict(checks: Checks) -> None:
    for target, low, high in (("dissolved_oxygen", 0.0, 25.0), ("hsi", 0.0, 1.0)):
        response = checks.post("/predict", {"target": target, "state": STATE})
        if response.status_code != 200:
            checks.record(FAIL, f"predict {target}", f"HTTP {response.status_code}")
            continue
        body = response.json()
        value = body["prediction"]
        in_range = low <= value <= high
        has_caveats = bool(body["caveats"])
        checks.record(PASS if in_range and has_caveats else FAIL,
                      f"predict {target}",
                      f"{value:.3f} {body['unit']}, {len(body['caveats'])} caveat(s)")


def check_batch(checks: Checks) -> None:
    response = checks.post("/batch_predict",
                           {"target": "dissolved_oxygen", "states": [STATE] * 10})
    if response.status_code != 200:
        return checks.record(FAIL, "batch predict", f"HTTP {response.status_code}")
    body = response.json()
    ok = body["n"] == 10 and len(set(body["predictions"])) == 1
    checks.record(PASS if ok else FAIL, "batch predict",
                  f"{body['n']} rows, identical states agree: "
                  f"{len(set(body['predictions'])) == 1}")


def check_explain(checks: Checks) -> None:
    response = checks.post("/explain", {"target": "dissolved_oxygen", "state": STATE})
    if response.status_code != 200:
        return checks.record(FAIL, "explain", f"HTTP {response.status_code}")

    body = response.json()
    total = sum(c["contribution"] for c in body["contributions"])
    gap = body["prediction"] - body["baseline"]

    # An all-zero explanation also "adds up", and that is exactly what a SHAP
    # explainer returns when its background is the row being explained. Check
    # the contributions are not degenerate before checking they balance.
    moved = any(abs(c["contribution"]) > 1e-6 for c in body["contributions"])
    checks.record(PASS if moved else FAIL, "SHAP contributions are non-zero",
                  "every contribution is 0 - the explainer has no background"
                  if not moved else
                  f"{sum(1 for c in body['contributions'] if c['contribution'])} "
                  "feature(s) contribute")
    checks.record(PASS if abs(total - gap) < 0.05 else FAIL,
                  "SHAP contributions add up",
                  f"sum {total:+.3f} vs gap {gap:+.3f}")
    checks.record(PASS if body["collinear_pairs"] else FAIL,
                  "collinear pairs disclosed",
                  f"{len(body['collinear_pairs'])} pair(s)")


def check_scenario(checks: Checks) -> None:
    response = checks.post("/scenario_run", {
        "scenario_name": "drought", "target": "dissolved_oxygen",
        "baseline": STATE, "modifications": {"discharge": -0.6},
    })
    if response.status_code != 200:
        return checks.record(FAIL, "scenario", f"HTTP {response.status_code}")

    body = response.json()
    warns = any("not re-run" in c or "does not re-run" in c for c in body["caveats"])
    checks.record(PASS if warns else FAIL, "scenario declares it is not a simulation",
                  f"{body['change']:+.3f} {body['unit']}")


def check_error_handling(checks: Checks) -> None:
    """A malformed request must come back as JSON, not an HTML error page."""
    response = checks.post("/predict", {"target": "dissolved_oxygen", "state": {}})
    is_json = response.headers.get("content-type", "").startswith("application/json")
    checks.record(PASS if response.status_code == 422 and is_json else FAIL,
                  "empty state rejected as JSON",
                  f"HTTP {response.status_code}, {response.headers.get('content-type')}")


def check_latency(checks: Checks, rounds: int = 20) -> None:
    payload = {"target": "dissolved_oxygen", "state": STATE}
    checks.post("/predict", payload)  # warm

    started = time.perf_counter()
    for _ in range(rounds):
        checks.post("/predict", payload)
    per_call = (time.perf_counter() - started) * 1000 / rounds
    checks.record(PASS, "warm prediction latency", f"{per_call:.1f} ms per call")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000",
                        help="base URL of the running API")
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
            check_provenance(checks)
            check_predict(checks)
            check_batch(checks)
            check_explain(checks)
            check_scenario(checks)
            check_latency(checks)
        else:
            checks.record(SKIP, "model-dependent checks",
                          "no models loaded; mount ./data or run train_models.py")

    print(f"\n{checks.failures} failure(s), {checks.skipped} skipped")
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())
