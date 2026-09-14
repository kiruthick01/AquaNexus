"""Fabricate a stand-in artefact set, so a container can be checked with models in it.

The real artefacts are build outputs of 9.5 GB of LAS tiles, a Windows-only
HEC-RAS run and a monitoring record that is not redistributable, so CI has none
of them. Every container check therefore ran against a degraded API: it proved
the image starts and answers /health, and could not prove that a container with
artefacts mounted actually loads them, explains a prediction, or interpolates a
scenario. That is the half of the deployment that fails silently - see the
DATA_DIR note in the Dockerfile for how it fails.

This writes artefacts of the right *shape* from fabricated inputs, by feeding
them through the real training code in ``train_models.py``. The schema therefore
cannot drift from the real manifest: it is written by the same functions.

What it is NOT: a model of anything. The hydraulics come from power laws, not
HEC-RAS, and the chemistry is drawn from a seeded generator, not measured. Every
number it produces is meaningless as science. It exists so that ``docker run``
can be checked against a service that is loaded rather than degraded, and it
labels itself so that a reader cannot mistake one for the other:

- ``manifest["stand_in"]`` is true, which the real manifest never carries.
- Every model's first caveat begins STAND-IN ARTEFACT.
- It refuses to write over existing artefacts unless given --force.

Where it writes is ``DATA_DIR``, which the image already pins to /app/data. Point
it somewhere else through the environment, not through a flag: every path is
derived when the settings are constructed, and the trainers this reuses hold
their own reference to them.

    DATA_DIR=/tmp/aqdata python scripts/make_stand_in_artifacts.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("scripts.stand_in")

STAND_IN_CAVEAT = (
    "STAND-IN ARTEFACT. Trained on fabricated hydraulics and fabricated "
    "chemistry to give a container something to load. Every number it predicts "
    "is meaningless; it says nothing about the Ayase or about any river."
)

#: Cross-sections in the corrected geometry, so the reach means average over the
#: same count the real dataset does.
N_SECTIONS = 49
#: Observations for the water quality model, across four stations.
N_OBSERVATIONS = 144
#: Observations broadcast across sections for the habitat model. Fewer, because
#: this one is 49 rows per observation and an xgboost fit in CI is not free.
N_STATE_OBSERVATIONS = 24


def load_trainers():
    """Import the real trainers from ``train_models.py``.

    By path, because ``scripts/`` is a directory of entry points rather than a
    package. The point of importing them at all is that the manifest this writes
    is built by the same code as the real one, so a field added there appears
    here without anybody remembering to copy it.
    """
    path = Path(__file__).resolve().parent / "train_models.py"
    spec = importlib.util.spec_from_file_location("train_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fabricate_sweep(rng: np.random.Generator) -> pd.DataFrame:
    """A flow sweep shaped like the HEC-RAS output, from at-a-station hydraulics.

    Depth, velocity and width follow the usual power laws in discharge. The
    exponents are textbook values, not calibrated ones - this has to be
    interpolable and monotone, not right.
    """
    discharges = np.geomspace(0.2, 70.0, 12)
    stations = np.linspace(27500.0, 0.0, N_SECTIONS)

    rows = []
    for station in stations:
        # Downstream sections sit lower and run wider.
        invert = 2.0 + 9.0 * (station / 27500.0)
        width_scale = 40.0 + 20.0 * (1.0 - station / 27500.0)
        roughness = 1.0 + 0.05 * rng.standard_normal()
        for discharge in discharges:
            depth = 0.35 * discharge**0.40 * roughness
            velocity = 0.18 * discharge**0.30 * roughness
            top_width = width_scale * discharge**0.10
            rows.append({
                "discharge_bc": float(discharge),
                "river_station": float(station),
                "wse": float(invert + depth),
                "invert": float(invert),
                "depth": float(depth),
                "velocity": float(velocity),
                "discharge": float(discharge),
                "flow_area": float(depth * top_width),
                "top_width": float(top_width),
                "energy_slope": float(0.0004 * velocity**2 / max(depth, 0.05)),
            })
    return pd.DataFrame(rows)


def fabricate_observations(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Monthly grab samples from four stations, with a seasonal signal in them.

    The signal matters. A model fitted on noise predicts the mean for every
    input; its SHAP explanation then comes back all zeros, and the smoke check
    that looks for non-degenerate contributions can no longer tell a broken
    explainer from a flat model.
    """
    timestamps = pd.date_range("2022-04-30", periods=n, freq="ME")
    month = timestamps.month.to_numpy(dtype=float)
    season = np.sin(2 * np.pi * (month - 3) / 12.0)

    water_temp = 16.0 + 9.0 * season + rng.normal(0, 1.0, n)
    air_temp = water_temp + 2.0 + rng.normal(0, 1.5, n)
    discharge = np.exp(rng.normal(np.log(6.0), 0.8, n)).clip(0.2, 70.0)
    # Colder water holds more oxygen, and more flow reaerates: the two terms the
    # real model finds, put in on purpose so there is something to recover.
    dissolved_oxygen = (12.0 - 0.20 * water_temp + 0.35 * np.log(discharge)
                        + rng.normal(0, 0.7, n)).clip(1.0, 18.0)

    return pd.DataFrame({
        "timestamp": timestamps,
        "station": [f"stand-in-{i % 4 + 1}" for i in range(n)],
        "water_body": settings.RIVER_NAME_JA,
        "discharge": discharge,
        "water_temp": water_temp,
        "air_temp": air_temp,
        "dissolved_oxygen": dissolved_oxygen,
        "ph": rng.normal(7.4, 0.25, n),
        "bod": np.abs(rng.normal(3.0, 1.0, n)),
        "cod": np.abs(rng.normal(7.0, 2.0, n)),
        "suspended_solids": np.abs(rng.normal(12.0, 5.0, n)),
        "nitrogen_total": np.abs(rng.normal(3.0, 0.8, n)),
        "phosphorus_total": np.abs(rng.normal(0.25, 0.08, n)),
    })


def existing_artefacts(models_dir: Path) -> list[str]:
    return sorted(path.name for path in models_dir.glob("*")
                  if path.suffix in {".joblib", ".json", ".csv"})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="overwrite artefacts that are already there")
    args = parser.parse_args()

    settings.ensure_dirs()
    present = existing_artefacts(settings.MODELS_DIR)
    if present and not args.force:
        log.error("%s already holds %s; refusing to overwrite trained artefacts "
                  "with stand-ins (pass --force if that is what you want)",
                  settings.MODELS_DIR, ", ".join(present))
        return 1

    sweep_path = settings.PROCESSED_DIR / "ayase_flow_sweep.csv"
    if sweep_path.is_file() and not args.force:
        log.error("%s already exists; refusing to overwrite it", sweep_path)
        return 1

    rng = np.random.default_rng(settings.RANDOM_SEED)
    sweep = fabricate_sweep(rng)
    observations = fabricate_observations(rng, N_OBSERVATIONS)

    sweep.to_csv(sweep_path, index=False)
    log.info("fabricated sweep: %d section(s) x %d discharge(s) -> %s",
             sweep["river_station"].nunique(), sweep["discharge_bc"].nunique(),
             sweep_path)

    trainers = load_trainers()
    manifest = {"river": settings.RIVER_NAME, "river_ja": settings.RIVER_NAME_JA,
                "crs_epsg": settings.CRS_EPSG, "stand_in": True, "models": {}}

    jobs = (
        ("dissolved_oxygen", trainers.train_dissolved_oxygen, observations),
        ("hsi", trainers.train_hsi, observations.head(N_STATE_OBSERVATIONS)),
    )
    for name, trainer, rows in jobs:
        path, meta = trainer(rows, sweep)
        meta["path"] = path.name
        meta["caveats"] = [STAND_IN_CAVEAT, *meta.get("caveats", [])]
        manifest["models"][name] = meta
        log.info("stand-in %s: %s", name, path)

    manifest_path = settings.MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    log.info("wrote %s", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
