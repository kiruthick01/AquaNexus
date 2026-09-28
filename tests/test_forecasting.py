"""Tests for the Phase 1 temporal infrastructure.

These run against synthetic regular and irregular series only - see
``src/aquanexus/ml/forecasting.py`` and ``docs/DATA_LIMITATIONS.md`` for why the
real dissolved-oxygen record cannot exercise this module meaningfully yet. What
matters here is that the density-awareness, no-shuffle, and leakage-detection
guarantees actually hold.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.ml.forecasting import (
    add_lag_features,
    add_lead_targets,
    add_rolling_features,
    assert_no_temporal_leakage,
    walk_forward_splits,
)


@pytest.fixture
def daily_series():
    """One station, 40 consecutive daily observations - dense enough for every
    lag/window used below."""
    n = 40
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "timestamp": dates,
        "station": ["A"] * n,
        "value": np.arange(n, dtype=float),
    })


@pytest.fixture
def monthly_grab_series():
    """One station, monthly grab samples - the real record's actual density."""
    n = 12
    dates = pd.date_range("2024-01-01", periods=n, freq="30D")
    return pd.DataFrame({
        "timestamp": dates,
        "station": ["A"] * n,
        "value": np.arange(n, dtype=float),
    })


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------


def test_lag_feature_matches_true_shift_on_dense_series(daily_series):
    out = add_lag_features(daily_series, "value", lags=(1, 3, 7))
    # On a perfectly regular daily series, lag N should equal value N rows back.
    expected_lag1 = daily_series["value"].shift(1).to_numpy()
    got_lag1 = out["value_lag1d"].to_numpy()
    np.testing.assert_allclose(got_lag1[1:], expected_lag1[1:])
    assert np.isnan(got_lag1[0])


def test_lag_feature_records_actual_gap(daily_series):
    out = add_lag_features(daily_series, "value", lags=(3,))
    gaps = out["value_lag3d_gap_days"].dropna()
    # Every accepted gap is within tolerance of the nominal 3 days; rows early in
    # the series (fewer than 3 days of history available) may be accepted at a
    # shorter gap, but rows with full history behind them hit it exactly.
    assert (np.abs(gaps.to_numpy() - 3.0) <= 1.0).all()
    assert np.allclose(gaps.to_numpy()[1:], 3.0)


def test_lag_feature_rejects_out_of_tolerance_gap(monthly_grab_series):
    # Nominal lag of 1 day on a ~30-day-spaced series: the nearest real
    # observation is ~30 days away, far outside a +/-20% (floored at 1 day)
    # tolerance around 1 day.
    out = add_lag_features(monthly_grab_series, "value", lags=(1,),
                           tolerance_fraction=0.2, min_tolerance_days=1.0)
    assert out["value_lag1d"].notna().sum() == 0


def test_lag_feature_never_uses_future_row(daily_series):
    shuffled = daily_series.sample(frac=1.0, random_state=0)
    out = add_lag_features(shuffled, "value", lags=(1,))
    merged = out.sort_values("timestamp")
    for i in range(1, len(merged)):
        lag_val = merged["value_lag1d"].iloc[i]
        if np.isfinite(lag_val):
            assert lag_val == merged["value"].iloc[i - 1]


# ---------------------------------------------------------------------------
# Lead targets (forecasting labels)
# ---------------------------------------------------------------------------


def test_lead_target_matches_true_future_value_on_dense_series(daily_series):
    out = add_lead_targets(daily_series, "value", horizons=(1, 3, 7))
    expected_lead1 = daily_series["value"].shift(-1).to_numpy()
    got_lead1 = out["value_lead1d"].to_numpy()
    np.testing.assert_allclose(got_lead1[:-1], expected_lead1[:-1])
    assert np.isnan(got_lead1[-1])


def test_lead_target_rejects_out_of_tolerance_gap(monthly_grab_series):
    out = add_lead_targets(monthly_grab_series, "value", horizons=(1,),
                           tolerance_fraction=0.2, min_tolerance_days=1.0)
    assert out["value_lead1d"].notna().sum() == 0


def test_lead_target_never_uses_past_row(daily_series):
    shuffled = daily_series.sample(frac=1.0, random_state=2)
    out = add_lead_targets(shuffled, "value", horizons=(1,))
    merged = out.sort_values("timestamp")
    for i in range(len(merged) - 1):
        lead_val = merged["value_lead1d"].iloc[i]
        if np.isfinite(lead_val):
            assert lead_val == merged["value"].iloc[i + 1]


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------


def test_rolling_mean_matches_manual_time_window(daily_series):
    out = add_rolling_features(daily_series, "value", window="5D", min_periods=1,
                               stats=("mean",))
    col = "value_roll5D_mean"
    # Last row's 5-day window on a daily series covers the last 5 values.
    last = out.iloc[-1]
    expected = daily_series["value"].iloc[-5:].mean()
    assert last[col] == pytest.approx(expected)


def test_rolling_features_do_not_use_future_values(daily_series):
    out = add_rolling_features(daily_series, "value", window="3D", min_periods=1,
                               stats=("max",))
    # The rolling max up to and including the first row must equal that row's value.
    assert out["value_roll3D_max"].iloc[0] == daily_series["value"].iloc[0]


# ---------------------------------------------------------------------------
# Walk-forward split
# ---------------------------------------------------------------------------


def test_walk_forward_splits_are_chronological(daily_series):
    folds = walk_forward_splits(daily_series, n_folds=4)
    assert len(folds) > 0
    stamps = pd.to_datetime(daily_series["timestamp"])
    for fold in folds:
        train_times = stamps[fold.train]
        test_times = stamps[fold.test]
        if len(train_times) and len(test_times):
            assert train_times.max() < test_times.min()


def test_walk_forward_never_shuffles(daily_series):
    shuffled = daily_series.sample(frac=1.0, random_state=3).reset_index(drop=True)
    folds = walk_forward_splits(shuffled, n_folds=4)
    for fold in folds:
        assert_no_temporal_leakage(shuffled, fold.train, fold.test)


def test_walk_forward_skips_groups_too_sparse_for_n_folds(monthly_grab_series):
    # 12 unique timestamps, 20 folds requested: cannot form 20 non-empty folds.
    folds = walk_forward_splits(monthly_grab_series, n_folds=20)
    assert folds == []


# ---------------------------------------------------------------------------
# Leakage detection
# ---------------------------------------------------------------------------


def test_leakage_detector_passes_valid_split(daily_series):
    train = (daily_series["timestamp"] < "2024-01-20").to_numpy()
    test = (daily_series["timestamp"] >= "2024-01-20").to_numpy()
    assert_no_temporal_leakage(daily_series, train, test)  # must not raise


def test_leakage_detector_catches_intentionally_leaky_split(daily_series):
    # Deliberately put one early row in "test" and one late row in "train".
    train = np.zeros(len(daily_series), dtype=bool)
    test = np.zeros(len(daily_series), dtype=bool)
    train[-1] = True   # latest timestamp in "train"
    test[0] = True      # earliest timestamp in "test"
    with pytest.raises(ValueError, match="leakage"):
        assert_no_temporal_leakage(daily_series, train, test)
