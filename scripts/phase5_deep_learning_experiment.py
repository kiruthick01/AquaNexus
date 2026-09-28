"""Phase 5 deep learning - MLP vs. the existing classical baselines, real data.

Runs the same leave-one-station-out protocol as every other model in this
project (``ml.validator.ModelValidator``) for Persistence, Ridge, Random
Forest, and XGBoost, then adds the MLP under an equivalent protocol: for each
held-out station, a further random validation slice is carved from the three
remaining training stations purely for early stopping (never touching the
held-out station), then the MLP predicts on the held-out station.

The LSTM is not run here - see ``src/aquanexus/ml/deep.py`` and
``docs/DEEP_LEARNING.md`` for why: Phase 1 found zero usable lag-complete
forecasting rows, so there is no real sequence for it to train on.

    python scripts/phase5_deep_learning_experiment.py
"""

from __future__ import annotations

import glob
import sys
import time

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.data.dataset import DO_FEATURES, DO_RANGE, build_water_quality_dataset
from aquanexus.data.loader import filter_stations, load_many
from aquanexus.logger import get_logger
from aquanexus.ml.deep import MLPModel
from aquanexus.ml.evaluator import evaluate
from aquanexus.ml.validator import ModelValidator

log = get_logger("scripts.phase5_deep_learning_experiment")

SEED = 42
VALIDATION_FRACTION = 0.2


def run_mlp(dataset: pd.DataFrame, features: list[str], target: str) -> dict:
    rng = np.random.default_rng(SEED)
    predictions = np.full(len(dataset), np.nan)
    best_epochs, param_counts, train_times = [], [], []

    for station in sorted(dataset["station"].unique()):
        train_mask = dataset["station"] != station
        test_mask = ~train_mask
        train = dataset[train_mask].reset_index(drop=True)
        test_idx = dataset.index[test_mask]

        n = len(train)
        shuffled = rng.permutation(n)
        n_val = max(5, int(round(n * VALIDATION_FRACTION)))
        val_idx, fit_idx = shuffled[:n_val], shuffled[n_val:]

        model = MLPModel(hidden_sizes=(16, 8), dropout=0.2, learning_rate=1e-2,
                         weight_decay=1e-3, max_epochs=300, patience=20, seed=SEED,
                         output_range=DO_RANGE)
        start = time.perf_counter()
        model.fit(train.iloc[fit_idx][features], train.iloc[fit_idx][target],
                 X_val=train.iloc[val_idx][features], y_val=train.iloc[val_idx][target])
        train_times.append(time.perf_counter() - start)
        best_epochs.append(model.curve.best_epoch)
        param_counts.append(model.n_parameters())

        predictions[dataset.index.get_indexer(test_idx)] = model.predict(
            dataset.loc[test_idx, features]
        )

    y_true = dataset[target].to_numpy()
    inference_start = time.perf_counter()
    model.predict(dataset[features])  # last fold's model, timing a full-dataset batch
    inference_time = time.perf_counter() - inference_start

    metrics = evaluate(y_true, predictions, "mlp", "station-held-out",
                      baseline=float(y_true.mean()))
    return {
        "model": "mlp", "n": metrics.n, "rmse": metrics.rmse, "mae": metrics.mae,
        "r2": metrics.r2, "bias": metrics.bias,
        "n_parameters": int(np.mean(param_counts)),
        "mean_best_epoch": float(np.mean(best_epochs)),
        "mean_train_seconds": float(np.mean(train_times)),
        "inference_seconds_per_138_rows": inference_time,
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

    classical = ModelValidator(dataset, features, target, "station").run(
        model_types=("linear", "random_forest", "xgboost"), output_range=DO_RANGE
    )
    classical_table = classical.table()

    mlp_result = run_mlp(dataset, features, target)

    print()
    print("=" * 78)
    print("PHASE 5 DEEP LEARNING vs. classical baselines (station-held-out CV)")
    print(classical_table[["model", "n", "rmse", "mae", "r2", "bias"]].round(3)
          .to_string(index=False))
    print(f"{'mlp':<16} {mlp_result['n']:>3} {mlp_result['rmse']:.3f}  "
          f"{mlp_result['mae']:.3f}  {mlp_result['r2']:.3f}  {mlp_result['bias']:.3f}")
    print("-" * 78)
    print(f"MLP: {mlp_result['n_parameters']} parameter(s), mean best epoch "
          f"{mlp_result['mean_best_epoch']:.0f}, mean train time "
          f"{mlp_result['mean_train_seconds']:.2f}s/fold, inference "
          f"{mlp_result['inference_seconds_per_138_rows']*1000:.1f}ms for "
          f"{len(dataset)} rows")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
