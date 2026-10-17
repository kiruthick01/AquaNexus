"""STAND-IN ARTEFACT generator: toy model bundles so a container can be smoke-tested.

    python scripts/make_stand_in_artifacts.py [--force]

CI has no trained models (they come from 565,000 downloaded samples), but a container
check is only worth running against a service that has loaded something. This writes
two bundles, ``geo`` and ``full``, fitted to a made-up temperature-only relationship,
plus a two-row station table. Every number they produce is meaningless and the bundles
say so in ``bundle.json``; never serve them to a user.

It refuses to write over an existing bundle directory unless ``--force`` is given, so
it cannot fabricate over a trained model on a machine that has one.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.data import features as feat
from aquanexus.ml import do_experiment as ex
from aquanexus.ml.do_bundle import DOBundle

NOTE = "STAND-IN ARTEFACT: toy model for smoke tests, not trained on monitoring data."


def toy_bundle(feature_set: str) -> DOBundle:
    cols = feat.FEATURE_SETS[feature_set]
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(500, len(cols))), columns=cols)
    frame["air_temp"] = rng.uniform(0, 30, 500)
    target = 12 - 0.2 * frame["air_temp"] + rng.normal(0, 0.3, 500)
    booster = ex.lgb.train(
        {**ex.DEFAULT_PARAMS, "num_leaves": 7, "min_child_samples": 5},
        ex.lgb.Dataset(frame, target),
        num_boost_round=30,
    )
    meta = {
        "stand_in": True,
        "note": NOTE,
        "low_do_weight": 1.0,
        "trained_on": {"source": "none (toy data)", "rows": 500},
        "held_out_strict": {},
        "held_out_note": NOTE,
        "importance": {"air_temp": 1.0},
        "training_ranges": {"air_temp": [0.5, 29.5]},
        "built": "stand-in",
        "commit": "stand-in",
    }
    return DOBundle(booster, cols, [6.0, 9.0], [1.5, 1.0, 0.8], 0.9, meta)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing bundles")
    args = parser.parse_args()

    out = settings.MODELS_DIR / "do"
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"refusing to overwrite {out}; pass --force to replace it")
        return 1
    for name, feature_set in (("geo", "geo"), ("full", "full")):
        toy_bundle(feature_set).save(out / name)
    pd.DataFrame(
        {
            "station_id": [1001, 1002],
            "pref_code": [11, 13],
            "latitude": [35.8, 35.7],
            "longitude": [139.8, 139.9],
            "elevation": [3.0, 1.0],
            "relief_10km": [5.0, 4.0],
            "rel_elevation": [0.5, -0.5],
            "last_date": pd.to_datetime(["2022-12-15", "2022-12-15"]),
            "last_do": [6.5, 3.2],
            "station_do_mean": [7.1, 4.0],
            "station_do_n": [120, 90],
            "name": ["stand-in A", "stand-in B"],
            "water_body": ["stand-in", "stand-in"],
        }
    ).to_parquet(out / "stations.parquet")
    print(f"wrote stand-in bundles to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
