"""Tests for the Phase 7 Bayesian models.

Small draw/tune/chain counts throughout - these check correctness of the
mechanism (shapes, interval ordering, the unseen-station-uses-hyperprior
rule), not converged inference quality, which the experiment script's larger
run and `docs/BAYESIAN_MODELING.md` are responsible for.

Skipped entirely if the ``bayesian`` optional extra (PyMC) is not installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")

from aquanexus.ml.bayesian import BayesianHierarchicalModel, BayesianLinearModel  # noqa: E402

SMALL = {"draws": 100, "tune": 100, "chains": 2, "seed": 0}


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(1)
    n = 60
    X = pd.DataFrame({"x1": rng.uniform(0, 10, n), "x2": rng.uniform(-5, 5, n)})
    y = pd.Series(2.0 * X["x1"] - 0.5 * X["x2"] + 3.0 + rng.normal(0, 0.5, n))
    return X, y


@pytest.fixture
def grouped_data():
    rng = np.random.default_rng(2)
    n_per_station = 20
    stations = ["A", "B", "C"]
    station_offsets = {"A": -1.0, "B": 0.0, "C": 1.5}
    rows = []
    for station in stations:
        x1 = rng.uniform(0, 10, n_per_station)
        y = 2.0 * x1 + station_offsets[station] + rng.normal(0, 0.3, n_per_station)
        for xi, yi in zip(x1, y, strict=True):
            rows.append({"x1": xi, "y": yi, "station": station})
    frame = pd.DataFrame(rows)
    return frame[["x1"]], frame["y"], frame["station"]


# ---------------------------------------------------------------------------
# BayesianLinearModel
# ---------------------------------------------------------------------------


def test_linear_fit_predict_shape(linear_data):
    X, y = linear_data
    model = BayesianLinearModel(**SMALL).fit(X, y)
    result = model.predict_interval(X, level=0.9)
    assert result.point.shape == (len(X),)
    assert np.all(result.lower <= result.point)
    assert np.all(result.point <= result.upper)


def test_linear_interval_widens_with_higher_level(linear_data):
    X, y = linear_data
    model = BayesianLinearModel(**SMALL).fit(X, y)
    narrow = model.predict_interval(X, level=0.5)
    wide = model.predict_interval(X, level=0.95)
    assert np.mean(wide.width()) > np.mean(narrow.width())


def test_linear_posterior_summary_has_expected_parameters(linear_data):
    X, y = linear_data
    model = BayesianLinearModel(**SMALL).fit(X, y)
    summary = model.posterior_summary(level=0.9)
    # One alpha, one beta per feature, one sigma.
    assert len(summary.table) == 1 + len(X.columns) + 1


def test_linear_raises_before_fit(linear_data):
    X, _ = linear_data
    with pytest.raises(RuntimeError):
        BayesianLinearModel().predict_interval(X)


def test_linear_clips_to_output_range(linear_data):
    X, y = linear_data
    model = BayesianLinearModel(output_range=(0.0, 10.0), **SMALL).fit(X, y)
    result = model.predict_interval(X, level=0.9)
    assert result.lower.min() >= 0.0
    assert result.upper.max() <= 10.0


# ---------------------------------------------------------------------------
# BayesianHierarchicalModel
# ---------------------------------------------------------------------------


def test_hierarchical_fit_predict_seen_station(grouped_data):
    X, y, groups = grouped_data
    model = BayesianHierarchicalModel(**SMALL).fit(X, y, groups)
    result = model.predict_interval(X[groups == "A"], level=0.9, station="A")
    assert result.point.shape == (int((groups == "A").sum()),)
    assert np.all(result.lower <= result.upper)


def test_hierarchical_unseen_station_uses_population_hyperprior(grouped_data):
    X, y, groups = grouped_data
    model = BayesianHierarchicalModel(**SMALL).fit(X, y, groups)
    # "D" was never in the training groups.
    x_new = pd.DataFrame({"x1": [5.0, 5.0, 5.0]})
    result_unseen = model.predict_interval(x_new, level=0.9, station="D")
    result_default = model.predict_interval(x_new, level=0.9, station=None)
    assert result_unseen.point.shape == (3,)
    # Both routes to "unseen" (explicit unknown name, and no name at all)
    # must use the same population-hyperprior mechanism, not by coincidence
    # produce wildly different intervals from two code paths that claim to
    # do the same thing.
    assert np.mean(result_unseen.width()) == pytest.approx(
        np.mean(result_default.width()), rel=0.5
    )


def test_hierarchical_seen_station_differs_from_unseen(grouped_data):
    X, y, groups = grouped_data
    model = BayesianHierarchicalModel(**SMALL).fit(X, y, groups)
    x_new = pd.DataFrame({"x1": [5.0]})
    # Station "C" has offset +1.5, well above the population mean - its own
    # fitted intercept should pull its prediction up relative to a fresh,
    # unseen-station draw from the shared population.
    seen = model.predict_interval(x_new, level=0.9, station="C").point[0]
    unseen = model.predict_interval(x_new, level=0.9, station="Z").point[0]
    assert seen > unseen


def test_hierarchical_raises_before_fit(grouped_data):
    X, _, _ = grouped_data
    with pytest.raises(RuntimeError):
        BayesianHierarchicalModel().predict_interval(X)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_linear_save_load_round_trip(linear_data, tmp_path):
    X, y = linear_data
    model = BayesianLinearModel(**SMALL).fit(X, y)
    before = model.predict_interval(X, level=0.9)

    path = model.save(tmp_path / "bayesian_linear.joblib")
    restored = BayesianLinearModel.load(path)
    after = restored.predict_interval(X, level=0.9)

    np.testing.assert_allclose(before.point, after.point)
    np.testing.assert_allclose(before.lower, after.lower)
    np.testing.assert_allclose(before.upper, after.upper)


def test_linear_posterior_summary_unavailable_after_load(linear_data, tmp_path):
    X, y = linear_data
    model = BayesianLinearModel(**SMALL).fit(X, y)
    path = model.save(tmp_path / "bayesian_linear.joblib")
    restored = BayesianLinearModel.load(path)
    with pytest.raises(RuntimeError, match="trace"):
        restored.posterior_summary()


def test_hierarchical_save_load_round_trip(grouped_data, tmp_path):
    X, y, groups = grouped_data
    model = BayesianHierarchicalModel(**SMALL).fit(X, y, groups)
    x_new = pd.DataFrame({"x1": [5.0, 6.0]})
    before_seen = model.predict_interval(x_new, level=0.9, station="A")
    before_unseen = model.predict_interval(x_new, level=0.9, station="Z")

    path = model.save(tmp_path / "bayesian_hierarchical.joblib")
    restored = BayesianHierarchicalModel.load(path)
    after_seen = restored.predict_interval(x_new, level=0.9, station="A")
    after_unseen = restored.predict_interval(x_new, level=0.9, station="Z")

    np.testing.assert_allclose(before_seen.point, after_seen.point)
    np.testing.assert_allclose(before_unseen.point, after_unseen.point)
