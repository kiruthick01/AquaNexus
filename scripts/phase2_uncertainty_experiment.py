"""Phase 2 predictive uncertainty - run against the real canonical dataset.

For each station held out in turn (the project's canonical protocol - see
``aquanexus.ml.validator.ModelValidator``), fits both uncertainty methods on
the remaining three stations and evaluates coverage/width on the held-out
one. This is a genuine out-of-sample evaluation: the held-out station never
enters training, residual estimation, or quantile regression for its own
fold.

    python scripts/phase2_uncertainty_experiment.py
"""

from __future__ import annotations

import glob
import sys
from collections.abc import Callable

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.data.dataset import DO_FEATURES, DO_RANGE, build_water_quality_dataset
from aquanexus.data.loader import filter_stations, load_many
from aquanexus.logger import get_logger
from aquanexus.ml.evaluator import evaluate
from aquanexus.ml.uncertainty import IntervalModel, IntervalPrediction, coverage, mean_width

log = get_logger("scripts.phase2_uncertainty_experiment")

LEVEL = 0.9
SEED = 42


def leave_one_station_out(dataset: pd.DataFrame):
    """Yields (station, train_frame, test_frame), the canonical protocol."""
    for station in sorted(dataset["station"].unique()):
        train = dataset[dataset["station"] != station].reset_index(drop=True)
        test = dataset[dataset["station"] == station].reset_index(drop=True)
        yield station, train, test


def evaluate_method(
    name: str,
    build_model: Callable[[], IntervalModel],
    dataset: pd.DataFrame,
    features: list[str],
    target: str,
    pass_groups: bool,
) -> dict:
    """Leave-one-station-out coverage/width/point-accuracy for one method."""
    rows, all_y, all_point, all_lower, all_upper = [], [], [], [], []

    for station, train, test in leave_one_station_out(dataset):
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]

        model = build_model()
        if pass_groups:
            model.fit(X_train, y_train, groups=train["station"])
        else:
            model.fit(X_train, y_train)
        result = model.predict_interval(X_test, level=LEVEL)

        rows.append({
            "station": station,
            "n_test": len(test),
            "coverage": coverage(y_test.to_numpy(), result),
            "mean_width": mean_width(result),
            "mae": float(np.mean(np.abs(result.point - y_test.to_numpy()))),
        })
        all_y.append(y_test.to_numpy())
        all_point.append(result.point)
        all_lower.append(result.lower)
        all_upper.append(result.upper)

    per_station = pd.DataFrame(rows)
    y_all = np.concatenate(all_y)
    pooled = IntervalPrediction(
        point=np.concatenate(all_point), lower=np.concatenate(all_lower),
        upper=np.concatenate(all_upper), level=LEVEL,
    )
    metrics = evaluate(y_all, pooled.point, name, "station-held-out",
                       baseline=float(y_all.mean()))

    log.info("%s per-station:\n%s", name, per_station.round(3).to_string(index=False))
    return {
        "coverage": coverage(y_all, pooled),
        "mean_width": mean_width(pooled),
        "rmse": metrics.rmse,
        "r2": metrics.r2,
        "mae": metrics.mae,
        "per_station": per_station,
    }


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

    from aquanexus.ml.uncertainty import BootstrapIntervalModel, QuantileIntervalModel

    results = {
        "bootstrap": evaluate_method(
            "bootstrap",
            lambda: BootstrapIntervalModel(model_type="linear", n_bootstrap=500,
                                           seed=SEED, output_range=DO_RANGE),
            dataset, features, target, pass_groups=True,
        ),
        "quantile": evaluate_method(
            "quantile",
            lambda: QuantileIntervalModel(level=LEVEL, seed=SEED),
            dataset, features, target, pass_groups=False,
        ),
    }

    print()
    print("=" * 70)
    print(f"PHASE 2 UNCERTAINTY - pooled station-held-out results (target {LEVEL:.0%} PI)")
    for name, summary in results.items():
        print(f"{name}: coverage={summary['coverage']:.3f}  "
              f"mean_width={summary['mean_width']:.3f}  RMSE={summary['rmse']:.3f}  "
              f"R2={summary['r2']:.3f}  MAE={summary['mae']:.3f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
