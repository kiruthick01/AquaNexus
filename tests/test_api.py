"""API tests.

These run against the real trained models when present and skip otherwise, so a
fresh clone without artefacts still gets a green suite. The provenance tests are
the important ones: the API serves one model trained on synthetic labels, and a
caller must be able to tell which is which from the response alone.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aquanexus.api.app import app  # noqa: E402
from aquanexus.api.registry import registry  # noqa: E402
from aquanexus.config import settings  # noqa: E402

models_present = pytest.mark.skipif(
    not (settings.MODELS_DIR / "manifest.json").is_file(),
    reason="trained models absent; run scripts/train_models.py",
)

STATE = {
    "water_temp": 24.5, "dissolved_oxygen": 7.2, "discharge": 12.0,
    "depth": 1.8, "velocity": 0.45, "suspended_solids": 15.0, "month": 7,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Meta
# ---------------------------------------------------------------------------


def test_root_describes_the_service(client):
    body = client.get("/").json()
    assert body["name"] == settings.PROJECT_NAME
    assert "綾瀬川" in body["river"]


def test_health_always_answers(client):
    """Health must report degraded rather than fail when models are missing."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "degraded"}


@models_present
def test_health_is_ok_with_models(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["models_loaded"]) == {"dissolved_oxygen", "hsi"}


@models_present
def test_models_endpoint_declares_label_provenance(client):
    """The whole point: a caller can see which model is synthetic."""
    by_target = {m["target"]: m for m in client.get("/models").json()}
    assert by_target["dissolved_oxygen"]["labels"] == "observed"
    assert by_target["hsi"]["labels"] == "SYNTHETIC"
    assert any("SYNTHETIC" in c for c in by_target["hsi"]["caveats"])


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


@models_present
@pytest.mark.parametrize("target", ["dissolved_oxygen", "hsi"])
def test_predict_returns_a_value_with_caveats(client, target):
    body = client.post("/predict", json={"state": STATE, "target": target}).json()
    assert body["target"] == target
    assert body["caveats"], "every prediction must carry its caveats"
    assert body["interpretation"]


@models_present
def test_predicted_oxygen_is_physically_plausible(client):
    body = client.post("/predict", json={"state": STATE,
                                         "target": "dissolved_oxygen"}).json()
    assert 0.0 <= body["prediction"] <= 25.0
    assert body["unit"] == "mg/L"


@models_present
def test_predicted_hsi_is_bounded(client):
    body = client.post("/predict", json={"state": STATE, "target": "hsi"}).json()
    assert 0.0 <= body["prediction"] <= 1.0


@models_present
def test_out_of_range_input_is_flagged_not_refused(client):
    """A model asked about unseen conditions should answer and say so."""
    extreme = {**STATE, "discharge": 4000.0}
    response = client.post("/predict", json={"state": extreme,
                                             "target": "dissolved_oxygen"})
    assert response.status_code == 200
    flagged = {w["feature"] for w in response.json()["out_of_range"]}
    assert "discharge" in flagged


@models_present
def test_partial_state_still_predicts(client):
    """Missing fields are imputed, not rejected - a grab sample is partial."""
    response = client.post("/predict", json={"state": {"water_temp": 20.0},
                                             "target": "dissolved_oxygen"})
    assert response.status_code == 200


def test_empty_state_is_rejected(client):
    response = client.post("/predict", json={"state": {}, "target": "dissolved_oxygen"})
    assert response.status_code == 422


def test_impossible_value_is_rejected_by_schema(client):
    response = client.post("/predict",
                           json={"state": {**STATE, "water_temp": 500.0},
                                 "target": "dissolved_oxygen"})
    assert response.status_code == 422


def test_unknown_target_is_rejected(client):
    response = client.post("/predict", json={"state": STATE, "target": "rainfall"})
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


@models_present
def test_batch_returns_one_prediction_per_state(client):
    states = [STATE, {**STATE, "water_temp": 31.0}, {**STATE, "water_temp": 8.0}]
    body = client.post("/batch_predict",
                       json={"states": states, "target": "dissolved_oxygen"}).json()
    assert body["n"] == len(states) == len(body["predictions"])


@models_present
def test_batch_reflects_the_temperature_gradient(client):
    """Colder water holds more oxygen; the batch should show it."""
    states = [{**STATE, "water_temp": t} for t in (8.0, 20.0, 31.0)]
    body = client.post("/batch_predict",
                       json={"states": states, "target": "dissolved_oxygen"}).json()
    assert body["predictions"][0] > body["predictions"][-1]


def test_empty_batch_is_rejected(client):
    assert client.post("/batch_predict",
                       json={"states": [], "target": "hsi"}).status_code == 422


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@models_present
def test_scenario_reports_baseline_and_change(client):
    body = client.post("/scenario_run", json={
        "scenario_name": "half_discharge", "baseline": STATE,
        "modifications": {"discharge": -0.5}, "target": "dissolved_oxygen",
    }).json()
    assert body["change"] == pytest.approx(
        body["scenario_prediction"] - body["baseline_prediction"]
    )
    assert body["applied"]["discharge"] == pytest.approx(STATE["discharge"] * 0.5)


@models_present
def test_scenario_states_it_is_not_a_simulation(client):
    """Callers must not read a scenario as a re-run of the hydraulic model."""
    body = client.post("/scenario_run", json={
        "scenario_name": "x", "baseline": STATE,
        "modifications": {"discharge": -0.2}, "target": "hsi",
    }).json()
    assert any("does not re-run" in c for c in body["caveats"])


@models_present
def test_scenario_absolute_overrides_fractional(client):
    body = client.post("/scenario_run", json={
        "scenario_name": "x", "baseline": STATE,
        "modifications": {"discharge": -0.5},
        "absolute_modifications": {"discharge": 40.0},
        "target": "dissolved_oxygen",
    }).json()
    assert body["applied"]["discharge"] == pytest.approx(40.0)


def test_scenario_rejects_a_field_absent_from_the_baseline(client):
    response = client.post("/scenario_run", json={
        "scenario_name": "x", "baseline": {"water_temp": 20.0},
        "modifications": {"discharge": -0.2}, "target": "hsi",
    })
    assert response.status_code == 422


def test_scenario_rejects_a_change_below_minus_one(client):
    response = client.post("/scenario_run", json={
        "scenario_name": "x", "baseline": STATE,
        "modifications": {"discharge": -1.5}, "target": "hsi",
    })
    assert response.status_code == 422


@models_present
def test_low_flow_scenario_carries_the_unreliability_caveat(client):
    """The model is known to misbehave at low flow; the caveat must travel."""
    body = client.post("/scenario_run", json={
        "scenario_name": "drought", "baseline": STATE,
        "modifications": {"discharge": -0.9}, "target": "dissolved_oxygen",
    }).json()
    assert any("low flow" in c.lower() for c in body["caveats"])


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------


@models_present
def test_explain_returns_contributions_for_every_feature(client):
    body = client.post("/explain", json={"state": STATE,
                                         "target": "dissolved_oxygen"}).json()
    model = registry.get("dissolved_oxygen")
    assert len(body["contributions"]) == len(model.features)
    magnitudes = [abs(c["contribution"]) for c in body["contributions"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


@models_present
def test_contributions_sum_to_the_prediction_gap(client):
    """SHAP's additivity property, which is what makes the numbers meaningful."""
    body = client.post("/explain", json={"state": STATE,
                                         "target": "dissolved_oxygen"}).json()
    total = sum(c["contribution"] for c in body["contributions"])
    assert total == pytest.approx(body["prediction"] - body["baseline"], abs=0.05)


def test_explain_rejects_an_empty_state(client):
    assert client.post("/explain", json={"state": {},
                                         "target": "hsi"}).status_code == 422
