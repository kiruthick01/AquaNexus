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


def _network():
    """Stations 1 and 2 are ~1 km apart; station 3 is hundreds of km away."""
    rows = [
        (1, 35.00, 139.00, "2020-01-20", 9.0),
        (1, 35.00, 139.00, "2020-01-25", 8.5),
        (2, 35.01, 139.00, "2020-01-10", 6.0),
        (2, 35.01, 139.00, "2020-01-20", 2.0),
        (3, 40.00, 141.00, "2020-01-20", 7.0),
    ]
    return pd.DataFrame(
        {
            "station_id": [r[0] for r in rows],
            "latitude": [r[1] for r in rows],
            "longitude": [r[2] for r in rows],
            "date": pd.to_datetime([r[3] for r in rows]),
            "hour": 10,
            "dissolved_oxygen": [r[4] for r in rows],
        }
    )


def test_neighbors_exclude_self_and_distant_stations():
    out = features.add_neighbors(_network())
    far = out[out["station_id"] == 3].iloc[0]
    assert far["nb_count"] == 0
    assert np.isnan(far["nb_do_prior"])
    near = out[out["station_id"] == 1]
    assert (near["nb_count"] == 1).all()


def test_neighbor_prior_uses_only_strictly_earlier_visits():
    out = features.add_neighbors(_network())
    # station 1 on Jan 20: neighbour 2 visited Jan 10 (6.0) before it, and Jan 20 itself
    # (2.0) does not count as earlier, so the prior is exactly 6.0
    jan20 = out[(out["station_id"] == 1) & (out["date"] == "2020-01-20")].iloc[0]
    assert jan20["nb_do_prior"] == pytest.approx(6.0)
    # on Jan 25 the neighbour's running mean covers both visits: (6 + 2) / 2
    jan25 = out[(out["station_id"] == 1) & (out["date"] == "2020-01-25")].iloc[0]
    assert jan25["nb_do_prior"] == pytest.approx(4.0)


def test_neighbor_nowcast_respects_the_window():
    out = features.add_neighbors(_network(), window_days=10)
    jan20 = out[(out["station_id"] == 1) & (out["date"] == "2020-01-20")].iloc[0]
    assert jan20["nb_do_now"] == pytest.approx(2.0)  # same-day neighbour visit
    tight = features.add_neighbors(_network(), window_days=2)
    jan25 = tight[(tight["station_id"] == 1) & (tight["date"] == "2020-01-25")].iloc[0]
    assert np.isnan(jan25["nb_do_now"])  # nearest neighbour visit is 5 days away


def test_a_stations_own_label_never_reaches_its_own_neighbor_features():
    base = features.add_neighbors(_network())
    tampered = _network()
    tampered.loc[tampered["station_id"] == 1, "dissolved_oxygen"] = 999.0
    after = features.add_neighbors(tampered)
    cols = features.NEIGHBOR_PRIOR + features.NEIGHBOR_NOW
    own = base["station_id"] == 1
    pd.testing.assert_frame_equal(base.loc[own, cols], after.loc[own, cols])


def test_terrain_features_use_neighbouring_stations_for_relief():
    from aquanexus.data import geography

    stations = pd.DataFrame(
        {
            "station_id": [1, 2, 3, 4],
            "latitude": [35.00, 35.02, 35.04, 40.0],
            "longitude": [139.0, 139.0, 139.0, 141.0],
        }
    )
    elevations = pd.DataFrame({"station_id": [1, 2, 3, 4], "elevation": [10.0, 40.0, 100.0, 5.0]})
    out = geography.terrain_features(elevations, stations).set_index("station_id")
    # stations 1-3 lie within ~4.5 km of each other; station 4 is alone
    assert out.loc[1, "relief_10km"] == 90.0
    assert out.loc[2, "rel_elevation"] == pytest.approx(40.0 - 50.0)
    assert out.loc[3, "rel_elevation"] == pytest.approx(100.0 - 50.0)
    assert np.isnan(out.loc[4, "relief_10km"])
    assert out.loc[4, "elevation"] == 5.0


def test_a_restricted_pool_keeps_excluded_stations_out_of_every_neighbour_feature():
    network = _network()
    pool = (network["station_id"] != 2).to_numpy()
    out = features.add_neighbors(network, pool=pool)
    near = out[out["station_id"] == 1]
    assert near["nb_do_prior"].isna().all()
    assert near["nb_do_now"].isna().all()
    assert (near["nb_count"] == 1).all()  # density still counts the excluded station
