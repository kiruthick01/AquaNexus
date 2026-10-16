"""Model-ready features for nationwide dissolved-oxygen prediction.

Three feature sets, because what a model may know changes what its score means:

``env``
    Season, location and weather only. Usable anywhere weather and coordinates are
    known, including a station never sampled before. This is the headline set.
``history``
    ``env`` plus what the same station read on its previous visits (last DO, a
    running station mean, days since the last visit). A genuine forecast for a
    monitored station, and unavailable for a new one, so it is reported separately.
``concurrent``
    ``env`` plus pH, BOD and SS measured in the same sample. These are cheap field
    readings, so this is a soft sensor for DO, not a forecast. Also kept separate.

History features use only earlier visits of the same station, never the current
row, so they cannot leak the label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.data import moe, weather

TARGET = "dissolved_oxygen"

SEASON = ["doy_sin", "doy_cos", "month"]
LOCATION = ["latitude", "longitude"]
WEATHER = [
    "air_temp",
    "air_temp_3d",
    "air_temp_7d",
    "air_temp_30d",
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "rain_14d",
    "rain_30d",
    "days_since_rain",
    "solar",
    "wind",
]
HISTORY = ["prev_do", "prev_do_gap_days", "station_do_mean", "station_do_n"]
CONCURRENT = ["ph", "bod", "ss"]

FEATURE_SETS = {
    "env": SEASON + LOCATION + WEATHER,
    "history": SEASON + LOCATION + WEATHER + HISTORY,
    "concurrent": SEASON + LOCATION + WEATHER + CONCURRENT,
}


def add_season(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    doy = out["date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-station look-back features built only from strictly earlier visits.

    Rows must not be shuffled across stations before calling; they are sorted here.
    The running mean is an expanding mean shifted by one visit.
    """
    out = frame.sort_values(["station_id", "date", "hour"]).copy()
    grouped = out.groupby("station_id", sort=False)
    out["prev_do"] = grouped[TARGET].shift(1)
    out["prev_do_gap_days"] = (out["date"] - grouped["date"].shift(1)).dt.days
    expanding = grouped[TARGET].expanding().mean().reset_index(level=0, drop=True)
    out["station_do_mean"] = expanding.groupby(out["station_id"]).shift(1)
    out["station_do_n"] = grouped.cumcount()
    return out.reset_index(drop=True)


def build_frame(raw_dir: Path | str, weather_dir: Path | str) -> pd.DataFrame:
    """Samples + stations + weather + season + history, one row per DO sample."""
    samples = moe.load_samples(moe.find_files(raw_dir, "mk"))
    stations = moe.load_stations(moe.find_files(raw_dir, "mm"))
    frame = moe.build_dataset(samples, stations)
    frame = weather.join_weather(frame, weather_dir)
    frame = add_season(frame)
    return add_history(frame)


def usable(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rows with every feature present.

    Weather windows are NaN for the first 30 days of the series and for cells with
    no cached weather; history features are NaN on a station's first visit. LightGBM
    handles missing values natively, so callers that use it should skip this.
    """
    return frame.dropna(subset=features).reset_index(drop=True)
