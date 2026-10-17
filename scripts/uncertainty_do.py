"""Conformal intervals and SHAP drivers for the strict-regime DO models.

    python scripts/uncertainty_do.py

Train stations are split three ways (fit / early-stopping / calibration) so the
conformal residuals come from stations the model never fitted. Intervals are scored
on the strict test set: stations unseen in training, fiscal years after the purge.
Coverage is reported overall and where it is hardest: the low-DO tail and by season.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.data import features as feat
from aquanexus.logger import get_logger
from aquanexus.ml import do_experiment as ex

log = get_logger("scripts.uncertainty_do")

SEASONS = {
    **dict.fromkeys([12, 1, 2], "winter"),
    **dict.fromkeys([3, 4, 5], "spring"),
    **dict.fromkeys([6, 7, 8], "summer"),
    **dict.fromkeys([9, 10, 11], "autumn"),
}


def carve_stations(train: pd.DataFrame, frac: float, seed: int):
    rng = np.random.default_rng(seed)
    stations = np.sort(train["station_id"].unique())
    chosen = set(rng.choice(stations, size=int(len(stations) * frac), replace=False))
    mask = train["station_id"].isin(chosen).to_numpy()
    return train[~mask], train[mask]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("data/processed/do_frame.parquet"))
    parser.add_argument("--sets", nargs="*", default=["nowcast", "full"])
    parser.add_argument("--level", type=float, default=0.9)
    parser.add_argument("--shap-rows", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=Path("experiments"))
    args = parser.parse_args()

    import shap

    frame = pd.read_parquet(args.cache)
    split = next(s for s in ex.make_splits(frame) if s.name == "strict")
    train, test = frame[split.train], frame[split.test].reset_index(drop=True)
    fit, calibration = carve_stations(train, 0.1, seed=1)
    log.info("fit %d, calibration %d, test %d rows", len(fit), len(calibration), len(test))

    rows, groups = [], []
    for name in args.sets:
        cols = feat.FEATURE_SETS[name]
        model = ex.fit_lgbm(fit, cols)
        model.save_model(f"data/models/do_strict_{name}.txt")
        cal_pred = ex.predict_lgbm(model, calibration, cols)
        pred = ex.predict_lgbm(model, test, cols)
        y = test[feat.TARGET].to_numpy()
        cal_y = calibration[feat.TARGET].to_numpy()

        lo_c, hi_c = ex.conformal_interval(model, calibration, test, cols, args.level)
        lo_b, hi_b = ex.binned_conformal_interval(cal_pred, cal_y, pred, level=args.level)
        for kind, lo, hi in [("constant", lo_c, hi_c), ("binned", lo_b, hi_b)]:
            rows.append({"features": name, "interval": kind, "nominal": args.level}
                        | ex.coverage(y, lo, hi))  # fmt: skip
            masks = {
                "all": np.ones(len(y), bool),
                "do_below_5": y < 5,
                "do_5_to_8": (y >= 5) & (y < 8),
                "do_8_plus": y >= 8,
            }
            season = test["month"].map(SEASONS).to_numpy()
            masks |= {f"season_{s}": season == s for s in ("winter", "spring", "summer", "autumn")}
            for group, m in masks.items():
                record = {"features": name, "interval": kind, "group": group, "n": int(m.sum())}
                groups.append(record | ex.coverage(y[m], lo[m], hi[m]))

        sample = test.sample(min(args.shap_rows, len(test)), random_state=0)
        values = shap.TreeExplainer(model).shap_values(sample[cols])
        importance = pd.DataFrame(
            {"feature": cols, "mean_abs_shap": np.abs(values).mean(axis=0)}
        ).sort_values("mean_abs_shap", ascending=False)
        importance.to_csv(args.out / f"do_shap_{name}.csv", index=False)
        log.info("%s top drivers:\n%s", name, importance.head(8).round(3).to_string(index=False))

    pd.DataFrame(rows).to_csv(args.out / "do_uncertainty.csv", index=False)
    table = pd.DataFrame(groups)
    table.to_csv(args.out / "do_uncertainty_groups.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(table[table.features == "full"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
