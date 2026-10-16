"""Tests for weather snapping and antecedent-window features (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from aquanexus.data import weather


def _series(rain, start="2022-01-01"):
    dates = pd.date_range(start, periods=len(rain), freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "air_temp": [10.0 + i for i in range(len(rain))],
            "rain": rain,
            "solar": 1.0,
            "wind": 2.0,
        }
    )


def test_snap_lands_on_the_power_grid():
    assert weather.snap_lat(35.9) == 36.0
    assert weather.snap_lon(139.7) == 140.0  # 139.7 / 0.625 = 223.5, nearest centre is 224 * 0.625
    assert weather.cell_key(35.9, 139.7) == "36.000_140.000"


def test_nearby_stations_share_a_cell():
    frame = pd.DataFrame({"latitude": [35.88, 35.95], "longitude": [139.70, 139.80]})
    assert weather.station_cells(frame)["cell"].nunique() == 1


def test_rain_windows_exclude_the_sampling_day():
    # 40 dry days, then 20 mm on day 41: a sample on day 41 must not see it.
    rain = [0.0] * 40 + [20.0] + [0.0] * 5
    features = weather.daily_features(_series(rain)).set_index("date")
    day41 = features.loc[pd.Timestamp("2022-01-01") + pd.Timedelta(days=40)]
    day42 = features.loc[pd.Timestamp("2022-01-01") + pd.Timedelta(days=41)]
    assert day41["rain_1d"] == 0.0
    assert day41["rain_30d"] == 0.0
    assert day42["rain_1d"] == pytest.approx(20.0)
    assert day42["rain_7d"] == pytest.approx(20.0)


def test_windows_are_nan_until_enough_history():
    features = weather.daily_features(_series([1.0] * 40))
    assert features["rain_30d"].iloc[:30].isna().all()
    assert features["rain_30d"].iloc[30] == pytest.approx(30.0)


def test_days_since_rain_counts_dry_days_and_resets():
    rain = [0.0, 5.0, 0.0, 0.0, 0.0]
    since = weather.daily_features(_series(rain))["days_since_rain"].tolist()
    # day i looks at rain up to day i-1; the 5 mm fell on day 1, so day 2 has 0 dry days
    assert since[2:] == [0, 1, 2]


def test_join_weather_attaches_features_by_cell_and_date(tmp_path):
    key = weather.cell_key(35.9, 139.7)
    _series([2.0] * 60).to_csv(tmp_path / f"{key}.csv", index=False)
    samples = pd.DataFrame(
        {
            "station_id": [1, 2],
            "latitude": [35.9, 24.4],
            "longitude": [139.7, 124.2],
            "date": pd.to_datetime(["2022-02-15", "2022-02-15"]),
            "hour": [10, 10],
        }
    )
    out = weather.join_weather(samples, tmp_path).set_index("station_id")
    assert out.loc[1, "rain_7d"] == pytest.approx(14.0)
    assert pd.isna(out.loc[2, "rain_7d"])  # no cached weather for that cell
