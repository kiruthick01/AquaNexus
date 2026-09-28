"""Tests for the Phase 2 prediction-interval infrastructure.

These check the interface contracts (interval brackets the point prediction,
width is positive, mismatched levels are rejected) and the two documented
degenerate cases (too few groups for out-of-fold residuals; quantile
crossing) rather than asserting a specific numeric coverage - with n~100
synthetic rows, exact coverage is itself noisy, which is the finding Phase 2
reports about the real dataset, not something to hide behind a loose test.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.ml.uncertainty import (
    BootstrapIntervalModel,
    IntervalPrediction,
    QuantileIntervalModel,
    coverage,
    mean_width,
)


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(7)
    n = 120
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    noise = rng.normal(0, 1.0, n)
    y = 2.0 * x1 - 0.5 * x2 + 3.0 + noise
    groups = pd.Series(np.tile(["A", "B", "C", "D"], n // 4))
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y), groups


# ---------------------------------------------------------------------------
# BootstrapIntervalModel
# ---------------------------------------------------------------------------


def test_bootstrap_interval_brackets_point(linear_data):
    X, y, groups = linear_data
    model = BootstrapIntervalModel(n_bootstrap=25, seed=0).fit(X, y, groups=groups)
    result = model.predict_interval(X, level=0.9)
    assert np.all(result.lower <= result.point)
    assert np.all(result.point <= result.upper)
    assert np.all(result.width() > 0)


def test_bootstrap_interval_widens_with_higher_level(linear_data):
    X, y, groups = linear_data
    model = BootstrapIntervalModel(n_bootstrap=25, seed=0).fit(X, y, groups=groups)
    narrow = model.predict_interval(X, level=0.5)
    wide = model.predict_interval(X, level=0.95)
    assert mean_width(wide) > mean_width(narrow)


def test_bootstrap_falls_back_with_too_few_groups(linear_data):
    X, y, _ = linear_data
    single_group = pd.Series(["only"] * len(X))
    # Must not raise, despite too few groups for GroupKFold.
    model = BootstrapIntervalModel(n_bootstrap=10, seed=0).fit(X, y, groups=single_group)
    result = model.predict_interval(X, level=0.9)
    assert len(result.point) == len(X)


def test_bootstrap_without_groups_uses_in_sample_residuals(linear_data):
    X, y, _ = linear_data
    model = BootstrapIntervalModel(n_bootstrap=10, seed=0).fit(X, y)
    result = model.predict_interval(X, level=0.9)
    assert len(result.point) == len(X)


def test_bootstrap_raises_before_fit(linear_data):
    X, _, _ = linear_data
    with pytest.raises(RuntimeError):
        BootstrapIntervalModel().predict_interval(X)


# ---------------------------------------------------------------------------
# QuantileIntervalModel
# ---------------------------------------------------------------------------


def test_quantile_interval_brackets_point_on_most_rows(linear_data):
    X, y, _ = linear_data
    model = QuantileIntervalModel(level=0.9).fit(X, y)
    result = model.predict_interval(X, level=0.9)
    # Quantile crossing is a documented, logged possibility, not asserted away -
    # but on this well-behaved synthetic set it should be rare.
    bracketed = (result.lower <= result.point) & (result.point <= result.upper)
    assert bracketed.mean() > 0.9


def test_quantile_interval_rejects_mismatched_level(linear_data):
    X, y, _ = linear_data
    model = QuantileIntervalModel(level=0.9).fit(X, y)
    with pytest.raises(ValueError, match="level"):
        model.predict_interval(X, level=0.5)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def test_coverage_counts_fraction_inside():
    interval = IntervalPrediction(
        point=np.array([1.0, 2.0, 3.0]),
        lower=np.array([0.5, 1.5, 10.0]),
        upper=np.array([1.5, 2.5, 11.0]),
        level=0.9,
    )
    y_true = np.array([1.0, 2.0, 3.0])  # third point falls outside [10, 11]
    assert coverage(y_true, interval) == pytest.approx(2 / 3)


def test_mean_width_matches_manual_average():
    interval = IntervalPrediction(
        point=np.array([1.0, 2.0]),
        lower=np.array([0.0, 1.0]),
        upper=np.array([2.0, 4.0]),
        level=0.9,
    )
    assert mean_width(interval) == pytest.approx((2.0 + 3.0) / 2)
