"""Phase 7 Bayesian modeling - pooled and hierarchical, real data.

Runs the same leave-one-station-out protocol as every other uncertainty
method in this project (`ModelValidator`, Phase 2/3's experiment scripts), so
the resulting coverage/width numbers are directly comparable to
`docs/UNCERTAINTY.md`'s bootstrap and conformal results without recomputing
either.

    python scripts/phase7_bayesian_experiment.py
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.data.dataset import DO_FEATURES, DO_RANGE, build_water_quality_dataset
from aquanexus.data.loader import filter_stations, load_many
from aquanexus.logger import get_logger
from aquanexus.ml.bayesian import BayesianHierarchicalModel, BayesianLinearModel
from aquanexus.ml.evaluator import evaluate
from aquanexus.ml.uncertainty import IntervalPrediction, coverage, mean_width

log = get_logger("scripts.phase7_bayesian_experiment")

LEVEL = 0.9
SEED = 42
SAMPLE_KWARGS = {"draws": 1000, "tune": 1000, "chains": 4, "seed": SEED}


def leave_one_station_out(dataset: pd.DataFrame):
    for station in sorted(dataset["station"].unique()):
        train = dataset[dataset["station"] != station].reset_index(drop=True)
        test = dataset[dataset["station"] == station].reset_index(drop=True)
        yield station, train, test


def run_pooled(dataset: pd.DataFrame, features: list[str], target: str) -> dict:
    rows, all_y, all_point, all_lower, all_upper = [], [], [], [], []
    for station, train, test in leave_one_station_out(dataset):
        model = BayesianLinearModel(output_range=DO_RANGE, **SAMPLE_KWARGS).fit(
            train[features], train[target]
        )
        result = model.predict_interval(test[features], level=LEVEL)
        y_test = test[target].to_numpy()
        rows.append({"station": station, "n": len(test),
                     "coverage": coverage(y_test, result), "mean_width": mean_width(result)})
        all_y.append(y_test)
        all_point.append(result.point)
        all_lower.append(result.lower)
        all_upper.append(result.upper)
    return _summarize("bayesian_linear (pooled)", rows, all_y, all_point, all_lower, all_upper)


def run_hierarchical(dataset: pd.DataFrame, features: list[str], target: str) -> dict:
    rows, all_y, all_point, all_lower, all_upper = [], [], [], [], []
    for station, train, test in leave_one_station_out(dataset):
        model = BayesianHierarchicalModel(output_range=DO_RANGE, **SAMPLE_KWARGS).fit(
            train[features], train[target], train["station"]
        )
        # `station` was never in `train`, so predict_interval correctly draws
        # a fresh intercept from the population hyperprior - see the class
        # docstring and docs/BAYESIAN_MODELING.md.
        result = model.predict_interval(test[features], level=LEVEL, station=station)
        y_test = test[target].to_numpy()
        rows.append({"station": station, "n": len(test),
                     "coverage": coverage(y_test, result), "mean_width": mean_width(result)})
        all_y.append(y_test)
        all_point.append(result.point)
        all_lower.append(result.lower)
        all_upper.append(result.upper)
    return _summarize("bayesian_hierarchical", rows, all_y, all_point, all_lower, all_upper)


def _summarize(name, rows, all_y, all_point, all_lower, all_upper) -> dict:
    per_station = pd.DataFrame(rows)
    y_all = np.concatenate(all_y)
    pooled = IntervalPrediction(point=np.concatenate(all_point), lower=np.concatenate(all_lower),
                                upper=np.concatenate(all_upper), level=LEVEL)
    metrics = evaluate(y_all, pooled.point, name, "station-held-out", baseline=float(y_all.mean()))
    log.info("%s per-station:\n%s", name, per_station.round(3).to_string(index=False))
    return {"name": name, "coverage": coverage(y_all, pooled), "mean_width": mean_width(pooled),
           "rmse": metrics.rmse, "r2": metrics.r2, "mae": metrics.mae}


def main() -> int:
    files = sorted(glob.glob(str(settings.RAW_DIR / "waterquality" / "saitama_*.xlsx")))
    if not files:
        log.error("no water quality files; run scripts/download_data.py first")
        return 1
    sweep_path = settings.PROCESSED_DIR / "ayase_flow_sweep.csv"
    if not sweep_path.is_file():
        log.error("no flow sweep at %s; run scripts/build_geometry.py first", sweep_path)
        return 1

    observations = filter_stations(load_many(files), water_body=settings.RIVER_NAME_JA)
    sweep = pd.read_csv(sweep_path)
    dataset = build_water_quality_dataset(observations, sweep)
    features = [f for f in DO_FEATURES if f in dataset.columns]
    target = "dissolved_oxygen"
    log.info("canonical dataset: %d rows, %d stations, %d feature(s)",
             len(dataset), dataset["station"].nunique(), len(features))

    pooled_result = run_pooled(dataset, features, target)
    hierarchical_result = run_hierarchical(dataset, features, target)

    print()
    print("=" * 78)
    print(f"PHASE 7 BAYESIAN MODELING - station-held-out (target {LEVEL:.0%} posterior "
          "predictive interval)")
    for result in (pooled_result, hierarchical_result):
        print(f"{result['name']}: coverage={result['coverage']:.3f}  "
              f"mean_width={result['mean_width']:.3f}  RMSE={result['rmse']:.3f}  "
              f"R2={result['r2']:.3f}  MAE={result['mae']:.3f}")
    print("-" * 78)
    print("For reference (not recomputed here, same protocol - docs/UNCERTAINTY.md):")
    print("  bootstrap:  coverage=0.920  mean_width=6.738  RMSE=1.794  R2=0.389")
    print("  quantile:   coverage=0.703  mean_width=3.411  RMSE=1.632  R2=0.494")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
