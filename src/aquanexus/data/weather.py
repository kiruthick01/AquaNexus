"""Daily weather for monitoring stations, from the NASA POWER daily API.

The MOE sample files carry no water temperature and no flow, so the two biggest
drivers of dissolved oxygen have to be inferred from weather: air temperature (a
proxy for water temperature, and for oxygen solubility), antecedent rainfall (a
proxy for flow and for runoff loading), solar radiation (algal production) and wind
(re-aeration).

Stations are snapped to POWER's native grid, 0.5 degrees of latitude by 0.625 of
longitude (about 55 km by 55 km at Japanese latitudes), and each cell is queried at
its centre. That is coarse: a local storm that misses the cell centre is invisible,
so the rain features are a regional signal, not a gauge reading. Each cell is
fetched once and cached as a CSV, so an interrupted download resumes where it
stopped. (Open-Meteo's ERA5 archive was tried first and abandoned: its free tier
weights a ten-year request so heavily that it allows about 47 cells an hour.)

Data: NASA Langley Research Center POWER Project, built on MERRA-2 and, for
precipitation, GPM IMERG. Open, no key. Cite "NASA POWER Project" in any write-up.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("data.weather")

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
LAT_STEP = 0.5
LON_STEP = 0.625
FILL_VALUE = -999.0

DAILY_VARIABLES = {
    "T2M": "air_temp",
    "PRECTOTCORR": "rain",
    "ALLSKY_SFC_SW_DWN": "solar",
    "WS10M": "wind",
}

# Antecedent windows end the day *before* the sample, so rain that fell after the
# sample was taken on the same day cannot leak into its features.
RAIN_WINDOWS = (1, 3, 7, 14, 30)
TEMP_WINDOWS = (3, 7, 30)


def snap_lat(value: float) -> float:
    return round(round(value / LAT_STEP) * LAT_STEP, 4)


def snap_lon(value: float) -> float:
    return round(round(value / LON_STEP) * LON_STEP, 4)


def cell_key(latitude: float, longitude: float) -> str:
    return f"{snap_lat(latitude):.3f}_{snap_lon(longitude):.3f}"


def station_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``cell`` (and its centre coordinates) to a frame with latitude/longitude."""
    out = frame.copy()
    out["cell_lat"] = out["latitude"].map(snap_lat)
    out["cell_lon"] = out["longitude"].map(snap_lon)
    out["cell"] = out["cell_lat"].map("{:.3f}".format) + "_" + out["cell_lon"].map("{:.3f}".format)
    return out


def _request(latitude: float, longitude: float, start: str, end: str) -> pd.DataFrame:
    query = urllib.parse.urlencode(
        {
            "parameters": ",".join(DAILY_VARIABLES),
            "community": "AG",
            "latitude": latitude,
            "longitude": longitude,
            "start": start.replace("-", ""),
            "end": end.replace("-", ""),
            "format": "JSON",
        }
    )
    with urllib.request.urlopen(f"{POWER_URL}?{query}", timeout=120) as response:
        payload = json.load(io.TextIOWrapper(response, encoding="utf-8"))
    series = payload["properties"]["parameter"]
    first = next(iter(series.values()))
    frame = pd.DataFrame({"date": pd.to_datetime(list(first), format="%Y%m%d")})
    for source, name in DAILY_VARIABLES.items():
        values = pd.Series(list(series[source].values()), dtype=float)
        frame[name] = values.replace(FILL_VALUE, float("nan"))
    return frame


def fetch_cell(
    latitude: float,
    longitude: float,
    start: str,
    end: str,
    cache_dir: Path | str,
    retries: int = 5,
) -> pd.DataFrame:
    """Daily weather for one grid cell, from the cache when present.

    The request is made at the cell centre. HTTP 429 means a rate limit; the wait
    grows with each attempt and the final failure is raised rather than swallowed,
    so a quota problem is never mistaken for a quiet gap in the data.
    """
    lat, lon = snap_lat(latitude), snap_lon(longitude)
    cache = Path(cache_dir) / f"{cell_key(lat, lon)}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    for attempt in range(retries):
        try:
            frame = _request(lat, lon, start, end)
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 503) or attempt == retries - 1:
                raise
            wait = 30 * 2**attempt
            log.warning("HTTP %d on %s, waiting %ds", error.code, cell_key(lat, lon), wait)
            time.sleep(wait)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False)
    return frame


def fetch_cells(
    cells: Iterable[tuple[float, float]],
    start: str,
    end: str,
    cache_dir: Path | str,
    pause: float = 0.5,
) -> int:
    """Fetch every missing cell. Returns the number of cells fetched this call."""
    fetched = 0
    cells = list(cells)
    for i, (lat, lon) in enumerate(cells, 1):
        if (Path(cache_dir) / f"{cell_key(lat, lon)}.csv").exists():
            continue
        fetch_cell(lat, lon, start, end, cache_dir)
        fetched += 1
        if fetched % 25 == 0:
            log.info("%d/%d cells (%d fetched this run)", i, len(cells), fetched)
        time.sleep(pause)
    return fetched


def daily_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Antecedent-window features for one cell's continuous daily series."""
    w = weather.sort_values("date").set_index("date").asfreq("D")
    out = pd.DataFrame(index=w.index)
    out["air_temp"] = w["air_temp"]
    out["solar"] = w["solar"]
    out["wind"] = w["wind"]
    prior_rain = w["rain"].shift(1)
    for days in RAIN_WINDOWS:
        out[f"rain_{days}d"] = prior_rain.rolling(days, min_periods=days).sum()
    prior_temp = w["air_temp"].shift(1)
    for days in TEMP_WINDOWS:
        out[f"air_temp_{days}d"] = prior_temp.rolling(days, min_periods=days).mean()
    # Days since the last wet day (>= 1 mm): a dry spell concentrates low-flow oxygen stress.
    wet = (prior_rain >= 1.0).to_numpy()
    since, run = [], 99
    for is_wet in wet:
        run = 0 if is_wet else min(run + 1, 99)
        since.append(run)
    out["days_since_rain"] = since
    return out.reset_index()


def join_weather(samples: pd.DataFrame, cache_dir: Path | str) -> pd.DataFrame:
    """Attach weather features to samples by grid cell and date.

    Rows whose cell has no cached weather, or whose window runs off the start of the
    cached series, keep NaN features; callers decide whether to drop them.
    """
    cached = {p.stem: p for p in Path(cache_dir).glob("*.csv")}
    frame = station_cells(samples)
    parts = []
    for cell, group in frame.groupby("cell"):
        if cell not in cached:
            parts.append(group)
            continue
        features = daily_features(pd.read_csv(cached[cell], parse_dates=["date"]))
        parts.append(group.merge(features, on="date", how="left"))
    out = pd.concat(parts, ignore_index=True).sort_values(["station_id", "date", "hour"])
    return out.reset_index(drop=True)
