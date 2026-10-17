"""API tests.

They run against a small bundle trained inside the test on a toy frame, so the suite
needs neither the real models nor the data. What they check is the contract: inputs
are assembled correctly, missing and unusual inputs are reported, intervals bracket
the prediction, and history is never taken from a date at or before the last record.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aquanexus.api.app import app  # noqa: E402
from aquanexus.api.registry import registry  # noqa: E402
from aquanexus.data import features as feat  # noqa: E402
from aquanexus.ml import do_experiment as ex  # noqa: E402
from aquanexus.ml.do_bundle import DOBundle  # noqa: E402

LAST = pd.Timestamp("2022-12-15")


def _toy_bundle(mode: str) -> DOBundle:
    cols = feat.FEATURE_SETS["geo" if mode != "full" else "full"]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(600, len(cols))), columns=cols)
    frame["air_temp"] = rng.uniform(0, 30, 600)
    y = 12 - 0.2 * frame["air_temp"] + rng.normal(0, 0.3, 600)
    booster = ex.lgb.train(
        {**ex.DEFAULT_PARAMS, "num_leaves": 7, "min_child_samples": 5},
        ex.lgb.Dataset(frame, y),
        num_boost_round=30,
    )
    classifier = None
    if mode == "geo":
        classifier = ex.lgb.train(
            {**ex.DEFAULT_PARAMS, "objective": "binary", "num_leaves": 7, "min_child_samples": 5},
            ex.lgb.Dataset(frame, (y < 5).astype(int)),
            num_boost_round=30,
        )
    meta = {
        "low_do_weight": 1.0,
        "trained_on": {"rows": 600},
        "held_out_strict": {"r2": 0.5},
        "held_out_note": "toy",
        "importance": {"air_temp": 1.0},
        "training_ranges": {"air_temp": [0.5, 29.5]},
        "built": "2026-01-01",
        "commit": "test",
    }
    return DOBundle(
        booster,
        cols,
        [6.0, 9.0],
        [1.5, 1.0, 0.8],
        0.9,
        meta,
        classifier,
        {"x": [0.0, 1.0], "y": [0.0, 1.0]} if classifier else None,
    )


@pytest.fixture
def client(tmp_path):
    for mode in ("geo", "full"):
        _toy_bundle(mode).save(tmp_path / mode)
    stations = pd.DataFrame(
        {
            "station_id": [1001, 1002],
            "pref_code": [11, 13],
            "latitude": [35.8, 35.7],
            "longitude": [139.8, 139.9],
            "elevation": [3.0, 1.0],
            "relief_10km": [5.0, 4.0],
            "rel_elevation": [0.5, -0.5],
            "last_date": [LAST, LAST],
            "last_do": [6.5, 3.2],
            "station_do_mean": [7.1, 4.0],
            "station_do_n": [120, 90],
            "name": ["綾瀬橋", "四ツ木橋"],
            "water_body": ["綾瀬川", "中川"],
        }
    )
    stations.to_parquet(tmp_path / "stations.parquet")
    registry.load(tmp_path)
    with TestClient(app) as c:
        registry.load(tmp_path)  # the app's startup load looks at the real models dir
        yield c
    registry.bundles.clear()
    registry.stations = None


def request(**over):
    base = {
        "date": "2023-07-10",
        "location": {"latitude": 35.8, "longitude": 139.8},
        "weather": {"air_temp": 25.0},
    }
    return base | over


def test_health_and_models(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["stations"] == 2
    models = {m["mode"]: m for m in client.get("/models").json()}
    assert set(models) == {"geo", "full"}
    assert models["geo"]["held_out_strict"]["r2"] == 0.5
    assert models["geo"]["interval_half_widths"] == [1.5, 1.0, 0.8]


def test_prediction_has_an_interval_that_brackets_it(client):
    body = client.post("/predict", json=request()).json()
    assert body["lower"] < body["prediction"] < body["upper"]
    assert body["level"] == 0.9
    assert body["caveats"]


def test_a_bundle_with_a_classifier_reports_a_low_oxygen_probability(client):
    hot = client.post("/predict", json=request(weather={"air_temp": 29.0})).json()
    cool = client.post("/predict", json=request(weather={"air_temp": 2.0})).json()
    assert 0 <= cool["probability_below_5"] <= hot["probability_below_5"] <= 1
    full = client.post(
        "/predict", json={"date": "2023-07-10", "station_id": 1001, "mode": "full"}
    ).json()
    assert full["probability_below_5"] is None


def test_warmer_air_predicts_lower_oxygen(client):
    cool = client.post("/predict", json=request(weather={"air_temp": 5.0})).json()
    warm = client.post("/predict", json=request(weather={"air_temp": 28.0})).json()
    assert warm["prediction"] < cool["prediction"]


def test_missing_inputs_are_listed_not_hidden(client):
    body = client.post("/predict", json=request()).json()
    assert "rain_7d" in body["missing_features"]
    assert "air_temp" not in body["missing_features"]


def test_out_of_range_input_is_flagged_not_refused(client):
    body = client.post("/predict", json=request(weather={"air_temp": 40.0})).json()
    flagged = {w["feature"] for w in body["out_of_range"]}
    assert flagged == {"air_temp"}


def test_station_fills_location_and_terrain(client):
    body = client.post("/predict", json={"date": "2023-07-10", "station_id": 1001}).json()
    assert "latitude" not in body["missing_features"]
    assert "elevation" not in body["missing_features"]


def test_history_is_used_only_after_the_last_record(client):
    after = client.post(
        "/predict", json={"date": "2023-07-10", "station_id": 1001, "mode": "full"}
    ).json()
    assert "prev_do" not in after["missing_features"]
    before = client.post(
        "/predict", json={"date": "2022-06-01", "station_id": 1001, "mode": "full"}
    )
    assert before.status_code == 422
    assert "history features were left out" in before.json()["detail"]


def test_a_model_that_needs_oxygen_readings_refuses_to_guess_without_any(client):
    response = client.post("/predict", json=request(mode="full"))
    assert response.status_code == 422
    assert "earlier or neighbouring oxygen readings" in response.json()["detail"]
    supplied = client.post("/predict", json=request(mode="full", history={"prev_do": 6.0}))
    assert supplied.status_code == 200


def test_contributions_sum_to_the_prediction(client):
    body = client.post("/predict", json=request(explain=True)).json()
    assert sum(body["contributions"].values()) == pytest.approx(body["prediction"], abs=1e-6)


def test_a_place_is_required(client):
    assert client.post("/predict", json={"date": "2023-07-10"}).status_code == 422


def test_impossible_value_is_rejected_by_the_schema(client):
    bad = request(weather={"air_temp": 90.0})
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_station_is_404(client):
    assert client.post("/predict", json={"date": "2023-07-10", "station_id": 5}).status_code == 404


def test_unloaded_mode_is_503(client):
    body = request(mode="nowcast")
    assert client.post("/predict", json=body).status_code == 503


def test_batch_returns_one_prediction_per_request(client):
    warm, cool = request(weather={"air_temp": 28.0}), request(weather={"air_temp": 5.0})
    body = client.post("/batch_predict", json={"requests": [warm, cool]}).json()
    assert body["n"] == 2
    assert body["predictions"][0]["prediction"] < body["predictions"][1]["prediction"]


def test_station_search(client):
    found = client.get("/stations", params={"q": "綾瀬"}).json()
    assert [s["station_id"] for s in found] == [1001]
    assert client.get("/stations", params={"prefecture": 13}).json()[0]["station_id"] == 1002
    assert client.get("/stations/1001").json()["name"] == "綾瀬橋"
    assert client.get("/stations/9").status_code == 404
