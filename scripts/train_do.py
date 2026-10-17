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
import gc
import json
from pathlib import Path

import pandas as pd

from aquanexus.data import features as feat
from aquanexus.data import moe
from aquanexus.logger import get_logger
from aquanexus.ml import do_experiment as ex

log = get_logger("scripts.train_do")


KEEP = ["station_id", "pref_code", "fiscal_year", "month", "date", feat.TARGET]


def slim(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Keep only the columns the run needs, with float32 features, to bound memory."""
    wanted = list(dict.fromkeys(KEEP + columns))
    out = frame[wanted].copy()
    floats = [c for c in out.select_dtypes("float64").columns if c != feat.TARGET]
    out[floats] = out[floats].astype("float32")
    return out


def load_frame(
    raw: Path,
    weather: Path,
    elevation: Path,
    cache: Path,
    rebuild: bool,
    landuse: Path | None = None,
):
    """Build (or load) the feature frame, keeping held-out stations out of neighbour pools.

    Excluding the stations that the ``station`` and ``strict`` regimes never train on
    means no training row's features contain a test station's reading.
    """
    if cache.exists() and not rebuild:
        return pd.read_parquet(cache)
    data = moe.build_dataset(
        moe.load_samples(moe.find_files(raw, "mk")), moe.load_stations(moe.find_files(raw, "mm"))
    )
    held = ex.held_out_stations(data["station_id"])
    frame = feat.build_frame(
        raw, weather, elevation, exclude_from_neighbors=held, landuse_cache=landuse
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache)
    return frame


def evaluate_regime(frame, split, sets, trials, rows, tail=()):
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

    for name, cols in sets.items():
        record("ridge", name, ex.ridge(train, test, cols))
        booster = ex.fit_lgbm(train, cols)
        record("lgbm_default", name, ex.predict_lgbm(booster, test, cols),
               {"rounds": booster.best_iteration})  # fmt: skip
        for weight in tail:
            booster = ex.fit_lgbm(train, cols, tail_weight=weight)
            record(f"lgbm_tail{weight:g}", name, ex.predict_lgbm(booster, test, cols),
                   {"rounds": booster.best_iteration})  # fmt: skip
        if trials:
            tuned = ex.tune_lgbm(train, cols, n_trials=trials)
            booster = ex.fit_lgbm(train, cols, tuned["params"])
            record("lgbm_tuned", name, ex.predict_lgbm(booster, test, cols),
                   {"rounds": booster.best_iteration, "cv_rmse": tuned["cv_rmse"],
                    "params": json.dumps(tuned["params"])})  # fmt: skip


def evaluate_prefectures(frame, sets, rows):
    """Pooled out-of-region predictions from five prefecture-held-out folds."""
    for name, cols in sets.items():
        preds = pd.Series(float("nan"), index=frame.index)
        for fold in ex.prefecture_folds(frame):
            booster = ex.fit_lgbm(frame[fold.train], cols)
            preds[fold.test] = ex.predict_lgbm(booster, frame[fold.test], cols)
        rows.append(
            {"regime": "prefecture", "model": "lgbm_default", "features": name}
            | ex.metrics(frame[feat.TARGET], preds.to_numpy())
        )


def save(rows: list[dict], out: Path) -> pd.DataFrame:
    """Write results so far, so an interrupted run keeps every finished regime."""
    table = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/moe"))
    parser.add_argument("--weather", type=Path, default=Path("data/cache/weather"))
    parser.add_argument("--elevation", type=Path, default=Path("data/cache/elevation.csv"))
    parser.add_argument("--landuse", type=Path, default=Path("data/cache/landuse.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/do_frame.parquet"))
    parser.add_argument("--sets", nargs="*", default=None, help="subset of feature sets")
    parser.add_argument("--out", type=Path, default=Path("experiments/do_results.csv"))
    parser.add_argument("--trials", type=int, default=0, help="Optuna trials per regime; 0 skips")
    parser.add_argument("--tail", type=float, nargs="*", default=(), help="low-DO sample weights")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--regimes", nargs="*", default=None, help="subset of regime names")
    parser.add_argument("--no-prefecture", action="store_true")
    args = parser.parse_args()

    sets = {k: v for k, v in feat.FEATURE_SETS.items() if not args.sets or k in args.sets}
    needed = sorted({c for cols in sets.values() for c in cols})
    full = load_frame(
        args.raw, args.weather, args.elevation, args.cache, args.rebuild, args.landuse
    )
    frame = slim(full, needed)
    del full
    log.info("%d rows, %d stations", len(frame), frame["station_id"].nunique())

    rows: list[dict] = []
    for split in ex.make_splits(frame):
        if args.regimes and split.name not in args.regimes:
            continue
        evaluate_regime(frame, split, sets, args.trials, rows, args.tail)
        save(rows, args.out)
        gc.collect()
    if not args.no_prefecture and (not args.regimes or "prefecture" in args.regimes):
        evaluate_prefectures(frame, sets, rows)

    table = save(rows, args.out)
    show = ["regime", "model", "features", "n", "r2", "mae", "rmse", "mae_low", "recall_low"]
    print(table[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
