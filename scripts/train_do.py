"""Train and evaluate the nationwide dissolved-oxygen model.

Runs every baseline and LightGBM across the evaluation regimes in
``aquanexus.ml.do_experiment`` and writes one results table, so each number in the
write-up traces to a row here.

    python scripts/train_do.py --trials 30

Tuning runs separately inside each regime, on that regime's training rows only, so
no regime's test rows ever inform another regime's hyperparameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aquanexus.data import features as feat
from aquanexus.logger import get_logger
from aquanexus.ml import do_experiment as ex

log = get_logger("scripts.train_do")


def load_frame(raw: Path, weather: Path, cache: Path, rebuild: bool) -> pd.DataFrame:
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    frame = feat.build_frame(raw, weather)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache)
    return frame


def evaluate_regime(frame, split, trials, rows):
    train, test = frame[split.train], frame[split.test]
    log.info("%s: train %d rows, test %d rows", split.name, len(train), len(test))

    def record(model, feature_set, pred, extra=None):
        rows.append(
            {"regime": split.name, "model": model, "features": feature_set}
            | ex.metrics(test[feat.TARGET], pred)
            | (extra or {})
        )

    record("global_mean", "-", ex.global_mean(train, test))
    record("climatology", "-", ex.climatology(train, test))
    record("persistence", "history", ex.persistence(test))

    for name, cols in feat.FEATURE_SETS.items():
        record("ridge", name, ex.ridge(train, test, cols))
        booster = ex.fit_lgbm(train, cols)
        record("lgbm_default", name, ex.predict_lgbm(booster, test, cols),
               {"rounds": booster.best_iteration})  # fmt: skip
        if trials:
            tuned = ex.tune_lgbm(train, cols, n_trials=trials)
            booster = ex.fit_lgbm(train, cols, tuned["params"])
            record("lgbm_tuned", name, ex.predict_lgbm(booster, test, cols),
                   {"rounds": booster.best_iteration, "cv_rmse": tuned["cv_rmse"],
                    "params": json.dumps(tuned["params"])})  # fmt: skip


def evaluate_prefectures(frame, rows):
    """Pooled out-of-region predictions from five prefecture-held-out folds."""
    for name, cols in feat.FEATURE_SETS.items():
        preds = pd.Series(float("nan"), index=frame.index)
        for fold in ex.prefecture_folds(frame):
            booster = ex.fit_lgbm(frame[fold.train], cols)
            preds[fold.test] = ex.predict_lgbm(booster, frame[fold.test], cols)
        rows.append(
            {"regime": "prefecture", "model": "lgbm_default", "features": name}
            | ex.metrics(frame[feat.TARGET], preds.to_numpy())
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/moe"))
    parser.add_argument("--weather", type=Path, default=Path("data/cache/weather"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/do_frame.parquet"))
    parser.add_argument("--out", type=Path, default=Path("experiments/do_results.csv"))
    parser.add_argument("--trials", type=int, default=0, help="Optuna trials per regime; 0 skips")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--regimes", nargs="*", default=None, help="subset of regime names")
    parser.add_argument("--no-prefecture", action="store_true")
    args = parser.parse_args()

    frame = load_frame(args.raw, args.weather, args.cache, args.rebuild)
    log.info("%d rows, %d stations", len(frame), frame["station_id"].nunique())

    rows: list[dict] = []
    for split in ex.make_splits(frame):
        if args.regimes and split.name not in args.regimes:
            continue
        evaluate_regime(frame, split, args.trials, rows)
    if not args.no_prefecture and not args.regimes:
        evaluate_prefectures(frame, rows)

    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    show = ["regime", "model", "features", "n", "r2", "mae", "rmse", "mae_low", "recall_low"]
    print(table[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
