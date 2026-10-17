"""Static station geography: elevation from GSI, and local slope from neighbouring stations.

Weather says what the sky did; it says nothing about which river a station sits on.
Per-station differences in dissolved oxygen (a steep cold headwater against a slow
lowland channel) are worth about a quarter of the variance, so each station gets
terrain descriptors:

* ``elevation``     - station height in metres, from GSI's national DEM (1-10 m
  resolution, laser-derived where available).
* ``relief_10km``   - highest minus lowest elevation among stations within 10 km,
  itself included. Stations sit on rivers, so this is largely the along-river drop.
* ``rel_elevation`` - station height minus the mean of those stations. Negative low
  in a catchment, positive high up.

Both relief measures are NaN for a station with no neighbour within 10 km; LightGBM
treats that as missing. GSI's elevation API is open (Japanese Government Standard
Terms of Use); requests are kept to four at a time with a short pause.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("data.geography")

GSI_URL = "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php"
EARTH_RADIUS_KM = 6371.0
RELIEF_RADIUS_KM = 10.0
WORKERS = 4


def station_pairs(stations: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    """Every ordered pair of distinct stations within ``radius_km`` of each other."""
    from scipy.spatial import cKDTree

    lat = np.radians(stations["latitude"].to_numpy())
    lon = np.radians(stations["longitude"].to_numpy())
    xyz = EARTH_RADIUS_KM * np.column_stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    )
    ids = stations["station_id"].to_numpy()
    found = cKDTree(xyz).query_pairs(radius_km, output_type="ndarray")
    a, b = found[:, 0], found[:, 1]
    dist = np.linalg.norm(xyz[a] - xyz[b], axis=1)
    return pd.DataFrame(
        {
            "station_id": np.concatenate([ids[a], ids[b]]),
            "neighbor_id": np.concatenate([ids[b], ids[a]]),
            "dist_km": np.concatenate([dist, dist]),
        }
    )


def _elevation(latitude: float, longitude: float, retries: int = 4) -> float:
    url = f"{GSI_URL}?lon={longitude:.6f}&lat={latitude:.6f}&outtype=JSON"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                value = json.load(response)["elevation"]
            return float(value) if not isinstance(value, str) else float("nan")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def fetch_elevation(stations: pd.DataFrame, cache: Path | str, chunk: int = 400) -> pd.DataFrame:
    """Elevation per station, cached and resumable.

    ``stations`` needs ``station_id``, ``latitude`` and ``longitude``. A station GSI
    cannot resolve (offshore, outside coverage) is stored as NaN rather than retried.
    """
    cache = Path(cache)
    empty = pd.DataFrame(columns=["station_id", "elevation"])
    done = pd.read_csv(cache) if cache.exists() else empty
    todo = stations[~stations["station_id"].isin(done["station_id"])].reset_index(drop=True)
    blocks = [done]
    for start in range(0, len(todo), chunk):
        part = todo.iloc[start : start + chunk]
        with ThreadPoolExecutor(WORKERS) as pool:
            values = list(pool.map(_elevation, part["latitude"], part["longitude"]))
        ids = part["station_id"].to_numpy()
        blocks.append(pd.DataFrame({"station_id": ids, "elevation": values}))
        out = pd.concat(blocks, ignore_index=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(cache, index=False)
        log.info("%d/%d stations", start + len(part), len(todo))
        time.sleep(1.0)
    return pd.concat(blocks, ignore_index=True)


def terrain_features(
    elevations: pd.DataFrame, stations: pd.DataFrame, radius_km: float = RELIEF_RADIUS_KM
) -> pd.DataFrame:
    """Elevation, local relief and relative elevation per station.

    ``elevations`` has ``station_id`` and ``elevation``; ``stations`` adds the
    coordinates used to find neighbours.
    """
    table = stations[["station_id", "latitude", "longitude"]].merge(elevations, on="station_id")
    pairs = station_pairs(table.reset_index(drop=True), radius_km)
    height = table.set_index("station_id")["elevation"]
    pairs["neighbor_elevation"] = pairs["neighbor_id"].map(height)
    near = pairs.groupby("station_id")["neighbor_elevation"].agg(["min", "max", "sum", "count"])

    own = table.set_index("station_id")["elevation"]
    both = near.reindex(own.index)
    top = np.fmax(both["max"], own)
    bottom = np.fmin(both["min"], own)
    mean = (both["sum"] + own) / (both["count"] + 1)
    has_neighbor = both["count"].fillna(0) > 0
    return pd.DataFrame(
        {
            "station_id": own.index.to_numpy(),
            "elevation": own.to_numpy(),
            "relief_10km": (top - bottom).where(has_neighbor).to_numpy(),
            "rel_elevation": (own - mean).where(has_neighbor).to_numpy(),
        }
    )
