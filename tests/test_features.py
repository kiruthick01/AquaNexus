"""Tests for season and station-history features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquanexus.data import features


def _frame():
    return pd.DataFrame(
        {
            "station_id": [1, 1, 1, 2, 2],
            "date": pd.to_datetime(
                ["2020-01-10", "2020-02-10", "2020-03-10", "2020-01-05", "2020-04-05"]
            ),
            "hour": [10] * 5,
            "dissolved_oxygen": [10.0, 8.0, 6.0, 9.0, 7.0],
        }
    )


def test_history_uses_only_earlier_visits():
    out = features.add_history(_frame())
    first = out[out["station_id"] == 1].reset_index(drop=True)
    assert np.isnan(first.loc[0, "prev_do"])
    assert first.loc[1, "prev_do"] == 10.0
    assert first.loc[2, "prev_do"] == 8.0
    # the running mean at visit 3 averages visits 1-2 only, never the current 6.0
    assert first.loc[2, "station_do_mean"] == pytest.approx(9.0)
    assert first["station_do_n"].tolist() == [0, 1, 2]


def test_history_gap_is_days_since_previous_visit():
    out = features.add_history(_frame())
    second = out[out["station_id"] == 2].reset_index(drop=True)
    assert np.isnan(second.loc[0, "prev_do_gap_days"])
    assert second.loc[1, "prev_do_gap_days"] == 91


def test_history_does_not_cross_stations():
    out = features.add_history(_frame())
    assert np.isnan(out[out["station_id"] == 2].iloc[0]["prev_do"])


def test_season_is_cyclic():
    frame = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-12-31"])})
    out = features.add_season(frame)
    assert out["doy_sin"].iloc[0] == pytest.approx(out["doy_sin"].iloc[1], abs=0.05)
    assert out["doy_cos"].iloc[0] == pytest.approx(out["doy_cos"].iloc[1], abs=0.01)


def test_target_and_labels_never_appear_in_any_feature_set():
    for columns in features.FEATURE_SETS.values():
        assert features.TARGET not in columns
        assert "below_standard" not in columns
