"""Train and save the deployable DO models, one bundle per feature set.

    python scripts/save_do_model.py

Each bundle is fitted on every fiscal year (FY2013-22) on 90% of stations; the other
10% of stations calibrate the conformal intervals, so the interval widths come from
stations the model never fitted. Bundles are written to ``data/models/do/<name>/``.
The accuracy figures stored in each bundle are the strict-regime held-out results
from ``experiments/`` (trained on FY<=2019, tested on unseen stations in FY2021-22),
not scores of the bundle itself, which has seen those years.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.data import features as feat
from aquanexus.logger import get_logger
from aquanexus.ml import do_experiment as ex
from aquanexus.ml.do_bundle import DOBundle

log = get_logger("scripts.save_do_model")

BINS = (6.0, 9.0)
# (bundle name, feature set, low-DO sample weight)
BUNDLES = [
    ("geo", "geo", 1.0),
    ("nowcast", "nowcast", 1.0),
    ("full", "full", 1.0),
    ("full_alert", "full", 10.0),
]


def commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def strict_results(results: Path, feature_set: str, weight: float) -> dict:
    """Held-out strict-regime numbers for a feature set, taken from the experiment CSVs."""
    model = "lgbm_default" if weight == 1.0 else f"lgbm_tail{weight:g}"
    frames = [pd.read_csv(p) for p in sorted(results.glob("do_results_*.csv"))]
    table = pd.concat(frames, ignore_index=True)
    row = table[
        (table["regime"] == "strict")
        & (table["features"] == feature_set)
        & (table["model"] == model)
    ]
    if row.empty:
        return {}
    keep = ["n", "r2", "mae", "rmse", "mae_low", "recall_low", "precision_low"]
    return {k: round(float(row.iloc[0][k]), 4) for k in keep if k in row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("data/processed/do_frame.parquet"))
    parser.add_argument("--results", type=Path, default=Path("experiments"))
    parser.add_argument("--out", type=Path, default=Path("data/models/do"))
    parser.add_argument("--level", type=float, default=0.9)
    args = parser.parse_args()

    frame = pd.read_parquet(args.cache)
    rng = np.random.default_rng(1)
    stations = np.sort(frame["station_id"].unique())
    cal_ids = set(rng.choice(stations, size=int(len(stations) * 0.1), replace=False))
    is_cal = frame["station_id"].isin(cal_ids).to_numpy()
    fit, calibration = frame[~is_cal], frame[is_cal]
    log.info("fit %d rows, calibration %d rows", len(fit), len(calibration))

    for name, feature_set, weight in BUNDLES:
        cols = feat.FEATURE_SETS[feature_set]
        booster = ex.fit_lgbm(fit, cols, tail_weight=weight)
        cal_pred = ex.predict_lgbm(booster, calibration, cols)
        cal_resid = np.abs(calibration[feat.TARGET].to_numpy() - cal_pred)
        cal_bin = np.digitize(cal_pred, BINS)
        widths = [ex.conformal_quantile(cal_resid[cal_bin == b], args.level) for b in range(3)]
        meta = {
            "name": name,
            "feature_set": feature_set,
            "low_do_weight": weight,
            "target": "dissolved oxygen, mg/L",
            "trained_on": {
                "source": "MOE public water body survey (river), FY2013-22",
                "rows": int(len(fit)),
                "stations": int(fit["station_id"].nunique()),
                "calibration_stations": len(cal_ids),
                "rounds": int(booster.best_iteration),
            },
            "held_out_strict": strict_results(args.results, feature_set, weight),
            "held_out_note": "Trained FY<=2019, scored on unseen stations in FY2021-22.",
            "built": date.today().isoformat(),
            "commit": commit(),
        }
        bundle = DOBundle(booster, cols, list(BINS), widths, args.level, meta)
        bundle.save(args.out / name)
        log.info("%s: %d rounds, interval half-widths %s", name, booster.best_iteration,
                 np.round(widths, 2).tolist())  # fmt: skip


if __name__ == "__main__":
    main()
