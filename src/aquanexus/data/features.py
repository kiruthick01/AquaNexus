"""Model-ready features for nationwide dissolved-oxygen prediction.

Feature sets, because what a model may know changes what its score means:

``env``
    Season, location and weather only. Usable anywhere weather and coordinates are
    known, including a station never sampled before.
``geo``
    ``env`` plus elevation and local relief (see :mod:`aquanexus.data.geography`).
    Still needs no monitoring at the site.
``spatial``
    ``geo`` plus what *other* stations within 25 km averaged on earlier visits. A
    prior for an unmonitored reach that has monitored neighbours.
``nowcast_basic``
    ``spatial`` plus what those neighbours read within 10 days of the sample. Needs a
    live neighbouring network, so it is a nowcast, not a forecast.
``nowcast``
    ``nowcast_basic`` plus the same neighbour summaries over stations within 5 km and over
    stations on a river of the same name within 60 km.
``history``
    ``geo`` plus what the same station read on earlier visits. A forecast for a
    monitored station, unavailable for a new one.
``full_basic``
    ``nowcast_basic`` plus ``history``.
``full``
    ``nowcast`` plus ``history``: everything causal, nothing measured in the sample.
``concurrent``
    ``geo`` plus pH, BOD and SS measured in the same sample. Cheap field readings,
    so this is a soft sensor for DO, not a forecast.

History and neighbour features use only observations of *other* visits (strictly
earlier for history and the prior, within a window for the nowcast) and never the
sample's own label, so none of them can leak it.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.data import geography, moe, weather

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
GEO = ["elevation", "relief_10km", "rel_elevation"]
HISTORY = ["prev_do", "prev_do_gap_days", "station_do_mean", "station_do_n"]
NEIGHBOR_PRIOR = ["nb_do_prior", "nb_do_prior_n", "nb_count", "nb_dist_mean"]
NEIGHBOR_NOW = ["nb_do_now", "nb_do_now_n"]
NEAR = ["n5_do_prior", "n5_do_prior_n", "n5_count", "n5_do_now", "n5_do_now_n"]
RIVER = ["rv_do_prior", "rv_do_prior_n", "rv_count", "rv_dist_mean", "rv_do_now", "rv_do_now_n"]
CONCURRENT = ["ph", "bod", "ss"]

_ENV = SEASON + LOCATION + WEATHER
FEATURE_SETS = {
    "env": _ENV,
    "geo": _ENV + GEO,
    "spatial": _ENV + GEO + NEIGHBOR_PRIOR,
    "nowcast_basic": _ENV + GEO + NEIGHBOR_PRIOR + NEIGHBOR_NOW,
    "nowcast": _ENV + GEO + NEIGHBOR_PRIOR + NEIGHBOR_NOW + NEAR + RIVER,
    "history": _ENV + GEO + HISTORY,
    "full_basic": _ENV + GEO + NEIGHBOR_PRIOR + NEIGHBOR_NOW + HISTORY,
    "full": _ENV + GEO + NEIGHBOR_PRIOR + NEIGHBOR_NOW + NEAR + RIVER + HISTORY,
    "concurrent": _ENV + GEO + CONCURRENT,
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


def _neighbor_stats(
    frame: pd.DataFrame,
    pairs: pd.DataFrame,
    source: pd.DataFrame,
    window_days: int,
    prefix: str,
) -> pd.DataFrame:
    """Neighbour summaries for every row of ``frame`` over the given station ``pairs``.

    ``source`` supplies the neighbour observations. All reads are of other stations,
    strictly earlier for the prior and within ``window_days`` for the nowcast.
    """
    visits = source.groupby(["station_id", "date"], as_index=False)[TARGET].mean()
    visits = visits.sort_values(["station_id", "date"])
    visits["running"] = visits.groupby("station_id")[TARGET].expanding().mean().to_numpy()
    obs = visits.rename(columns={"station_id": "neighbor_id", TARGET: "nb_do"})
    obs = obs.sort_values("date")

    left = frame[["station_id", "date"]].reset_index().rename(columns={"index": "row"})
    left = left.merge(pairs, on="station_id").sort_values("date")

    prior = pd.merge_asof(
        left,
        obs[["neighbor_id", "date", "running"]],
        on="date",
        by="neighbor_id",
        allow_exact_matches=False,
    )
    now = pd.merge_asof(
        left,
        obs[["neighbor_id", "date", "nb_do"]],
        on="date",
        by="neighbor_id",
        direction="nearest",
        tolerance=pd.Timedelta(days=window_days),
    )

    by_row = prior.groupby("row")
    out = pd.DataFrame(
        {
            f"{prefix}do_prior": by_row["running"].mean(),
            f"{prefix}do_prior_n": by_row["running"].count(),
            f"{prefix}count": by_row["neighbor_id"].nunique(),
            f"{prefix}dist_mean": by_row["dist_km"].mean(),
            f"{prefix}do_now": now.groupby("row")["nb_do"].mean(),
            f"{prefix}do_now_n": now.groupby("row")["nb_do"].count(),
        }
    ).reindex(frame.index)
    counts = [f"{prefix}count", f"{prefix}do_prior_n", f"{prefix}do_now_n"]
    out[counts] = out[counts].fillna(0)
    return out


_RIVER_SUFFIX = re.compile(r"(最下流|上流|中流|下流|河口|感潮域|本川|水域)$")
_PARENTHETICAL = re.compile(r"[（(].*?[）)]")


def river_key(name: object) -> str:
    """A river's name without segment marks (upstream/downstream, numbered reaches)."""
    text = _PARENTHETICAL.sub("", str(name)).strip()
    return _RIVER_SUFFIX.sub("", text)


def add_neighbors(
    frame: pd.DataFrame,
    radius_km: float = 25.0,
    window_days: int = 10,
    pool: pd.Series | np.ndarray | None = None,
    near_km: float = 5.0,
    river_km: float = 60.0,
) -> pd.DataFrame:
    """Neighbour-station features, never including the station's own readings.

    Three neighbourhoods, each with the same six columns under its own prefix:

    * ``nb_``: stations within ``radius_km``.
    * ``n5_``: stations within ``near_km``, which are usually the same reach.
    * ``rv_``: stations on a river of the same name within ``river_km`` (needs a
      ``water_body`` column), whatever their distance up to that limit.

    ``X_do_prior`` is the mean over neighbours of each neighbour's running mean DO as of
    its last visit strictly before the sample date; ``X_do_now`` is the mean of the
    visits nearest in time within ``window_days``; ``X_count`` counts neighbouring
    stations, a density proxy that uses no DO value.

    ``pool`` optionally restricts which rows may serve as neighbour observations (a
    boolean per row of ``frame``). Evaluation uses it to keep held-out stations' labels
    out of every feature, including the features of training rows.
    """
    stations = frame.drop_duplicates("station_id")[["station_id", "latitude", "longitude"]]
    stations = stations.reset_index(drop=True)
    wide = geography.station_pairs(stations, max(radius_km, river_km))
    source = frame if pool is None else frame[np.asarray(pool, dtype=bool)]

    blocks = [
        _neighbor_stats(frame, wide[wide["dist_km"] <= radius_km], source, window_days, "nb_"),
        _neighbor_stats(frame, wide[wide["dist_km"] <= near_km], source, window_days, "n5_"),
    ]
    if "water_body" in frame.columns:
        keys = frame.drop_duplicates("station_id").set_index("station_id")["water_body"]
        keys = keys.map(river_key)
        same = (
            keys.reindex(wide["station_id"]).to_numpy()
            == keys.reindex(wide["neighbor_id"]).to_numpy()
        )
        blocks.append(
            _neighbor_stats(
                frame, wide[same & (wide["dist_km"] <= river_km)], source, window_days, "rv_"
            )
        )
    return pd.concat([frame, *blocks], axis=1)


def build_frame(
    raw_dir: Path | str,
    weather_dir: Path | str,
    elevation_cache: Path | str | None = None,
    exclude_from_neighbors: set | None = None,
) -> pd.DataFrame:
    """Samples + stations + weather + season + terrain + neighbours + history.

    ``exclude_from_neighbors`` is a set of station ids whose readings may not feed any
    row's neighbour features; evaluation passes its held-out stations.
    """
    samples = moe.load_samples(moe.find_files(raw_dir, "mk"))
    stations = moe.load_stations(moe.find_files(raw_dir, "mm"))
    frame = moe.build_dataset(samples, stations)
    frame = weather.join_weather(frame, weather_dir)
    frame = add_season(frame)
    if elevation_cache is not None and Path(elevation_cache).exists():
        terrain = geography.terrain_features(
            pd.read_csv(elevation_cache), frame.drop_duplicates("station_id")
        )
        frame = frame.merge(terrain, on="station_id", how="left")
    pool = None if not exclude_from_neighbors else ~frame["station_id"].isin(exclude_from_neighbors)
    frame = add_neighbors(frame, pool=pool)
    return add_history(frame)


def usable(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Rows with every feature present.

    Weather windows are NaN for the first 30 days of the series and for cells with
    no cached weather; history features are NaN on a station's first visit. LightGBM
    handles missing values natively, so callers that use it should skip this.
    """
    return frame.dropna(subset=features).reset_index(drop=True)
