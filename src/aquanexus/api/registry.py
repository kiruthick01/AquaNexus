"""Loads the dissolved-oxygen bundles and the station table, and builds model inputs."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.data import features as feat
from aquanexus.logger import get_logger
from aquanexus.ml.do_bundle import DOBundle

log = get_logger("api.registry")

STATIONS_FILE = "stations.parquet"
STATION_FIELDS = ["latitude", "longitude", "elevation", "relief_10km", "rel_elevation"]
SITE_FIELDS = ["elevation", "relief_10km", "rel_elevation", *feat.LAND]
#: A point this close to a monitoring station borrows that station's terrain and land use.
NEAREST_STATION_KM = 3.0
HISTORY_FIELDS = ["prev_do", "prev_do_gap_days", "station_do_mean", "station_do_n"]
#: Features that carry an actual oxygen reading from somewhere else. Counts and
#: distances do not: they are available for any location without monitoring data.
SIGNAL_FEATURES = {
    c
    for c in feat.NEIGHBOR_PRIOR + feat.NEIGHBOR_NOW + feat.NEAR + feat.RIVER + feat.HISTORY
    if c.endswith(("do_prior", "do_now")) or c in {"prev_do", "station_do_mean"}
}

#: What each served model needs from the caller, for display next to the model.
MODE_SUMMARY = {
    "geo": "Coordinates, date, weather; terrain and land use from the nearest station.",
    "nowcast": "geo plus what neighbouring stations read in the last 10 days.",
    "full": "nowcast plus the station's own earlier readings. Most accurate.",
    "full_alert": "full, tuned to catch low-oxygen events (more false alarms).",
}


@dataclass
class BuiltInput:
    row: pd.DataFrame
    missing: list[str]
    out_of_range: list[dict]
    notes: list[str]


class ModelRegistry:
    """Holds the loaded bundles and the station table."""

    def __init__(self) -> None:
        self.bundles: dict[str, DOBundle] = {}
        self.stations: pd.DataFrame | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    @property
    def models(self) -> dict[str, DOBundle]:
        return self.bundles

    @property
    def ready(self) -> bool:
        return bool(self.bundles)

    def load(self, directory: Path | None = None) -> ModelRegistry:
        directory = Path(directory) if directory else settings.MODELS_DIR / "do"
        with self._lock:
            self.bundles.clear()
            self.stations, self.error = None, None
            if not directory.is_dir():
                self.error = f"no model directory at {directory}; run scripts/save_do_model.py"
                return self
            for path in sorted(p for p in directory.iterdir() if (p / "bundle.json").exists()):
                try:
                    self.bundles[path.name] = DOBundle.load(path)
                except Exception as exc:  # noqa: BLE001 - one bad bundle must not hide the rest
                    log.error("could not load %s: %s", path.name, exc)
            table = directory / STATIONS_FILE
            if table.exists():
                self.stations = pd.read_parquet(table).set_index("station_id", drop=False)
            if not self.bundles:
                self.error = f"no model bundles found in {directory}"
        return self

    def get(self, mode: str) -> DOBundle:
        if mode not in self.bundles:
            raise KeyError(f"model '{mode}' is not loaded (available: {sorted(self.bundles)})")
        return self.bundles[mode]

    def station(self, station_id: int) -> pd.Series | None:
        if self.stations is None or station_id not in self.stations.index:
            return None
        return self.stations.loc[station_id]

    def search_stations(self, text: str | None, prefecture: int | None, limit: int) -> pd.DataFrame:
        table = self.stations if self.stations is not None else pd.DataFrame()
        if table.empty:
            return table
        if prefecture is not None:
            table = table[table["pref_code"] == prefecture]
        if text:
            hit = table["name"].astype(str).str.contains(text, regex=False) | table[
                "water_body"
            ].astype(str).str.contains(text, regex=False)
            table = table[hit]
        return table.head(limit)

    def nearest_station(self, latitude, longitude) -> tuple[pd.Series, float] | None:
        """The closest monitoring station within ``NEAREST_STATION_KM``, with its distance."""
        if self.stations is None or latitude is None or longitude is None:
            return None
        lat = np.radians(self.stations["latitude"].to_numpy())
        lon = np.radians(self.stations["longitude"].to_numpy())
        la, lo = np.radians(latitude), np.radians(longitude)
        a = np.sin((lat - la) / 2) ** 2 + np.cos(la) * np.cos(lat) * np.sin((lon - lo) / 2) ** 2
        km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        i = int(np.argmin(km))
        return (self.stations.iloc[i], float(km[i])) if km[i] <= NEAREST_STATION_KM else None

    def build_input(self, bundle: DOBundle, request: dict) -> BuiltInput:
        """Assemble one feature row from a request.

        Station fields and history are filled from the station table when a
        ``station_id`` is given. History is only used when the requested date is
        after the station's last recorded visit; otherwise it would describe the
        future of the sample, so it is left missing and a note says so.
        """
        values: dict[str, float] = {}
        notes: list[str] = []
        day = pd.Timestamp(request["date"])

        row = self.station(request["station_id"]) if request.get("station_id") else None
        if request.get("station_id") and row is None:
            raise KeyError(f"unknown station_id {request['station_id']}")
        if row is not None:
            values.update({k: row[k] for k in STATION_FIELDS if k in row.index})
            values.update({k: row[k] for k in feat.LAND if k in row.index})
            if day > row["last_date"]:
                values.update(
                    prev_do=row["last_do"],
                    prev_do_gap_days=float((day - row["last_date"]).days),
                    station_do_mean=row["station_do_mean"],
                    station_do_n=row["station_do_n"],
                )
            else:
                notes.append(
                    f"date is not after the station's last record ({row['last_date']:%Y-%m-%d}); "
                    "its history features were left out"
                )

        for group in ("location", "weather", "history", "neighbors"):
            values.update({k: v for k, v in (request.get(group) or {}).items() if v is not None})

        if row is None:
            near = self.nearest_station(values.get("latitude"), values.get("longitude"))
            if near is not None:
                station, km = near
                borrowed = [
                    c
                    for c in SITE_FIELDS
                    if c in station.index and c in bundle.features and c not in values
                ]
                values.update({c: station[c] for c in borrowed})
                if borrowed:
                    notes.append(
                        f"terrain and land use taken from station {station['station_id']} "
                        f"({station['name']}), {km:.1f} km away"
                    )

        doy = day.dayofyear
        values.update(
            doy_sin=np.sin(2 * np.pi * doy / 365.25),
            doy_cos=np.cos(2 * np.pi * doy / 365.25),
            month=day.month,
        )
        frame = pd.DataFrame([{c: values.get(c, np.nan) for c in bundle.features}]).astype(float)
        missing = [c for c in bundle.features if pd.isna(frame.iloc[0][c])]
        ranges = bundle.metadata.get("training_ranges", {})
        out = [
            {"feature": c, "value": float(frame.iloc[0][c]), "training_min": lo, "training_max": hi}
            for c, (lo, hi) in ranges.items()
            if c in frame and not pd.isna(frame.iloc[0][c]) and not lo <= frame.iloc[0][c] <= hi
        ]
        return BuiltInput(frame, missing, out, notes)

    def missing_signal(self, bundle: DOBundle, built: BuiltInput) -> str | None:
        """Why a richer model cannot answer, or None if it has what it needs.

        ``nowcast`` and ``full`` learn from other readings of oxygen. If none of those
        arrive, the model falls back on how stations without neighbours look in the
        training data, which is a confident answer to a question nobody asked.
        """
        wanted = [c for c in bundle.features if c in SIGNAL_FEATURES]
        if not wanted:
            return None
        present = [c for c in wanted if c not in built.missing]
        if present:
            return None
        return (
            "this model needs earlier or neighbouring oxygen readings and none were supplied "
            f"(any of: {', '.join(wanted[:6])}, ...); give a station_id with a date after its last "
            "record, supply them, or use the 'geo' model"
        )

    def contributions(self, bundle: DOBundle, row: pd.DataFrame) -> dict[str, float]:
        """Exact TreeSHAP contributions in mg/L; they sum to the prediction minus ``bias``."""
        raw = bundle.booster.predict(
            row, num_iteration=bundle.booster.best_iteration, pred_contrib=True
        )
        names = [*bundle.features, "bias"]
        return {n: float(v) for n, v in zip(names, raw[0], strict=True)}


registry = ModelRegistry()

__all__ = ["BuiltInput", "ModelRegistry", "MODE_SUMMARY", "registry", "feat"]
