"""End-to-end integration tests (Phase 3b).

`test_api.py` checks each endpoint's contract. These check the seams between
them - that the artefacts on disk match what the manifest promises, that the
same state gets the same number from every endpoint that will answer about it,
that the process degrades honestly when the models are missing, and that the
error boundary returns JSON rather than an HTML traceback.

The seams are where this kind of service actually breaks: each endpoint can be
individually correct while `/predict` and `/explain` quietly disagree because
they build their features differently.
"""

from __future__ import annotations

import json
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aquanexus.api.app import app  # noqa: E402
from aquanexus.api.registry import ModelRegistry, registry  # noqa: E402
from aquanexus.config import settings  # noqa: E402

MANIFEST = settings.MODELS_DIR / "manifest.json"

models_present = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason="trained models absent; run scripts/train_models.py",
)

STATE = {
    "water_temp": 22.0, "dissolved_oxygen": 6.8, "discharge": 9.5,
    "depth": 1.6, "velocity": 0.4, "suspended_solids": 14.0, "month": 6,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture
def isolated_registry():
    """Restore the process-wide registry after a test disturbs it."""
    saved = (dict(registry.models), dict(registry.manifest), registry.error)
    yield registry
    registry.models, registry.manifest, registry.error = (
        saved[0], saved[1], saved[2])


# ---------------------------------------------------------------------------
# Artefacts on disk match what the manifest promises
# ---------------------------------------------------------------------------


@models_present
def test_every_manifest_model_loads(manifest):
    loaded = ModelRegistry().load()
    assert loaded.ready
    assert set(loaded.models) == set(manifest["models"])


@models_present
def test_manifest_features_match_the_saved_model(manifest):
    """A feature list that drifts from the artefact produces silent nonsense.

    `build_features` fills the manifest's feature names; the predictor selects
    its own. If they differ, the caller's input lands in the wrong column or
    goes missing entirely, and nothing raises.
    """
    loaded = ModelRegistry().load()
    for name, meta in manifest["models"].items():
        assert loaded.models[name].predictor.feature_names == meta["features"], name


@models_present
def test_training_ranges_cover_every_feature(manifest):
    """Out-of-range flagging is only as complete as the ranges recorded."""
    for name, meta in manifest["models"].items():
        missing = set(meta["features"]) - set(meta["training_ranges"])
        assert not missing, f"{name}: no training range for {sorted(missing)}"


@models_present
def test_collinear_pairs_name_real_features(manifest):
    """Regression: the pairs were promised by the schema and never populated.

    `/explain` documents `collinear_pairs` and returned an empty list for every
    request, because nothing ever wrote them. They are computed at training time
    now; a caller reading three separate hydraulic contributions as three
    findings is the failure this prevents.
    """
    for name, meta in manifest["models"].items():
        pairs = meta.get("collinear_pairs", [])
        assert pairs, f"{name}: no collinear pairs recorded"
        for a, b in pairs:
            assert a in meta["features"] and b in meta["features"], (name, a, b)


# ---------------------------------------------------------------------------
# The endpoints agree with each other
# ---------------------------------------------------------------------------


@models_present
@pytest.mark.parametrize("target", ["dissolved_oxygen", "hsi"])
def test_predict_explain_and_scenario_agree(client, target):
    """One state, three endpoints, one number."""
    predicted = client.post("/predict", json={"target": target, "state": STATE})
    explained = client.post("/explain", json={"target": target, "state": STATE})
    scenario = client.post("/scenario_run", json={
        "scenario_name": "no change", "target": target, "baseline": STATE,
        "modifications": {"discharge": 0.0},
    })
    assert predicted.status_code == explained.status_code == scenario.status_code == 200

    value = predicted.json()["prediction"]
    assert explained.json()["prediction"] == pytest.approx(value, rel=1e-6)
    assert scenario.json()["baseline_prediction"] == pytest.approx(value, rel=1e-6)
    assert scenario.json()["scenario_prediction"] == pytest.approx(value, rel=1e-6)


@models_present
def test_batch_matches_repeated_single_predictions(client):
    single = client.post("/predict", json={"target": "dissolved_oxygen",
                                           "state": STATE}).json()["prediction"]
    batch = client.post("/batch_predict", json={"target": "dissolved_oxygen",
                                                "states": [STATE] * 5}).json()
    assert batch["n"] == 5
    assert all(value == pytest.approx(single, rel=1e-6) for value in batch["predictions"])


@models_present
def test_inline_explanation_matches_the_explain_endpoint(client):
    """`/predict?explain=true` and `/explain` must not diverge."""
    inline = client.post("/predict", json={"target": "dissolved_oxygen",
                                           "state": STATE, "explain": True}).json()
    standalone = client.post("/explain", json={"target": "dissolved_oxygen",
                                               "state": STATE}).json()
    assert inline["contributions"] is not None

    dedicated = {c["feature"]: c["contribution"] for c in standalone["contributions"]}
    assert set(inline["contributions"]) == set(dedicated)
    for feature, value in inline["contributions"].items():
        assert value == pytest.approx(dedicated[feature], abs=1e-6), feature


@models_present
def test_explain_discloses_collinear_pairs(client, manifest):
    """The pairs reach the caller, not just the manifest."""
    body = client.post("/explain", json={"target": "dissolved_oxygen",
                                         "state": STATE}).json()
    served = {tuple(pair) for pair in body["collinear_pairs"]}
    recorded = {tuple(pair) for pair in
                manifest["models"]["dissolved_oxygen"]["collinear_pairs"]}
    assert served == recorded
    assert served, "explanation claims to disclose collinearity but discloses none"


@models_present
@pytest.mark.parametrize("target", ["dissolved_oxygen", "hsi"])
def test_explanations_are_not_degenerate(client, target):
    """Regression: /explain returned a contribution of zero for every feature.

    The explainer was fitted on the request row itself, so SHAP measured the
    prediction against a background of one identical point - baseline equal to
    prediction, every contribution exactly zero. The response looked
    well-formed, summed correctly, and said nothing. The background is now a
    sample of the training rows saved beside the model.
    """
    body = client.post("/explain", json={"target": target, "state": STATE}).json()

    contributions = [c["contribution"] for c in body["contributions"]]
    assert any(abs(c) > 1e-6 for c in contributions), "all contributions are zero"
    assert body["baseline"] != pytest.approx(body["prediction"], abs=1e-9), (
        "baseline equals the prediction; the background is the explained row"
    )
    assert sum(contributions) == pytest.approx(
        body["prediction"] - body["baseline"], abs=0.05)


@models_present
def test_inline_explanation_is_also_non_degenerate(client):
    body = client.post("/predict", json={"target": "dissolved_oxygen",
                                         "state": STATE, "explain": True}).json()
    assert any(abs(v) > 1e-6 for v in body["contributions"].values())


@models_present
def test_background_sample_ships_with_every_model(manifest):
    """The background is an artefact like the model; without it /explain is 503."""
    loaded = ModelRegistry().load()
    for name, meta in manifest["models"].items():
        assert meta.get("background"), f"{name}: no background recorded"
        assert (settings.MODELS_DIR / meta["background"]).is_file()

        model = loaded.models[name]
        assert model.background is not None
        assert list(model.background.columns) == meta["features"]


@models_present
def test_explainer_is_built_once_and_reused():
    """Fitting a KernelExplainer costs seconds; doing it per request is not viable."""
    loaded = ModelRegistry().load()
    model = loaded.models["dissolved_oxygen"]
    assert model.explainer is None

    first = loaded.explainer_for(model)
    assert first is not None
    assert loaded.explainer_for(model) is first


@models_present
def test_models_endpoint_matches_the_manifest(client, manifest):
    served = {m["target"]: m for m in client.get("/models").json()}
    assert set(served) == set(manifest["models"])
    for name, meta in manifest["models"].items():
        assert served[name]["features"] == meta["features"]
        assert served[name]["labels"] == meta["labels"]
        assert served[name]["n_train"] == meta["n_train"]


@models_present
def test_models_endpoint_publishes_training_ranges(client, manifest):
    """A client cannot show where the evidence ends unless it can see the edge.

    Without this the dashboard would have to hard-code the ranges, and they
    would drift the first time the models were retrained.
    """
    served = {m["target"]: m for m in client.get("/models").json()}
    for name, meta in manifest["models"].items():
        assert served[name]["training_ranges"], f"{name}: no ranges served"
        assert set(served[name]["training_ranges"]) == set(meta["training_ranges"])


@models_present
def test_out_of_range_check_follows_feature_aliases(client):
    """Regression: an aliased value reached the model unflagged.

    A caller supplies a point `depth`; `build_features` feeds it to the model as
    `reach_depth`, which is what the training range is recorded against. The
    check looked only for the literal name, so a depth far outside anything the
    model had seen was answered as though it were ordinary.
    """
    shallow = {**STATE, "depth": 0.2}
    body = client.post("/predict", json={"target": "dissolved_oxygen",
                                         "state": shallow}).json()

    flagged = {warning["feature"] for warning in body["out_of_range"]}
    assert "reach_depth" in flagged, body["out_of_range"]


@models_present
def test_a_state_inside_the_ranges_is_not_flagged(client):
    """The other half of it: no crying wolf on ordinary conditions."""
    ordinary = {**STATE, "depth": 2.9, "velocity": 0.41, "discharge": 12.0,
                "water_temp": 24.5}
    body = client.post("/predict", json={"target": "dissolved_oxygen",
                                         "state": ordinary}).json()
    assert body["out_of_range"] == []


# ---------------------------------------------------------------------------
# Degraded operation
# ---------------------------------------------------------------------------


def test_missing_models_degrade_rather_than_crash(tmp_path, isolated_registry):
    """No artefacts must not mean no service.

    The process still answers, /health says why, and the endpoints that need a
    model return 503 with the reason rather than a 500.
    """
    isolated_registry.models.clear()
    isolated_registry.error = None
    isolated_registry.load(models_dir=tmp_path)
    assert not isolated_registry.ready

    # TestClient's lifespan would reload the real models, so drive the app
    # without it and keep the empty registry in place.
    client = TestClient(app)

    health = client.get("/health").json()
    assert health["status"] == "degraded"
    assert health["models_loaded"] == []
    assert "manifest" in (health["detail"] or "")

    assert client.get("/").status_code == 200

    for path, payload in (
        ("/models", None),
        ("/predict", {"target": "dissolved_oxygen", "state": STATE}),
        ("/batch_predict", {"target": "dissolved_oxygen", "states": [STATE]}),
        ("/explain", {"target": "dissolved_oxygen", "state": STATE}),
        ("/scenario_run", {"scenario_name": "x", "target": "dissolved_oxygen",
                           "baseline": STATE, "modifications": {"discharge": -0.2}}),
    ):
        response = client.get(path) if payload is None else client.post(path, json=payload)
        assert response.status_code == 503, path
        assert response.json()["detail"], path


# ---------------------------------------------------------------------------
# Middleware and the error boundary
# ---------------------------------------------------------------------------


def test_every_response_is_timed(client):
    response = client.get("/health")
    assert float(response.headers["X-Response-Time-ms"]) >= 0


def test_cors_allows_the_configured_frontend_origin(client):
    origin = settings.CORS_ORIGINS[0]
    response = client.options("/predict", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
    })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_unhandled_errors_return_json_not_a_traceback(client, monkeypatch):
    """An HTML traceback page is useless to a client and leaks internals."""
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("aquanexus.api.routes.predictions.registry.get", explode)
    response = client.post("/predict", json={"target": "dissolved_oxygen",
                                             "state": STATE})
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["detail"] == "internal error"
    assert body["path"] == "/predict"
    assert "boom" not in response.text and "Traceback" not in response.text


# ---------------------------------------------------------------------------
# Documentation and startup cost
# ---------------------------------------------------------------------------


def test_openapi_documents_every_endpoint(client):
    schema = client.get("/openapi.json").json()
    assert client.get("/docs").status_code == 200
    for path in ("/health", "/models", "/predict", "/batch_predict",
                 "/explain", "/scenario_run"):
        assert path in schema["paths"], path


@models_present
def test_startup_and_prediction_stay_within_budget():
    """Loose ceilings: this catches an artefact that has become pathological.

    Measured on the development machine at roughly 60 ms to load both models and
    3 ms for a warm prediction; the bounds below are an order of magnitude
    looser, because a timing test that fails on a slow CI runner is a test that
    gets deleted.
    """
    started = time.perf_counter()
    ModelRegistry().load()
    startup_ms = (time.perf_counter() - started) * 1000
    assert startup_ms < 10_000, f"model loading took {startup_ms:.0f} ms"

    with TestClient(app) as client:
        payload = {"target": "dissolved_oxygen", "state": STATE}
        client.post("/predict", json=payload)  # warm the path

        started = time.perf_counter()
        for _ in range(10):
            assert client.post("/predict", json=payload).status_code == 200
        per_request_ms = (time.perf_counter() - started) * 100
        assert per_request_ms < 1_000, f"{per_request_ms:.0f} ms per prediction"
