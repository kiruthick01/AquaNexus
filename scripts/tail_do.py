"""The low-oxygen tail: a classifier for DO below 5 mg/L, and adaptive intervals.

    python scripts/tail_do.py --sets full nowcast

Strict regime (unseen stations, later years). Two questions:

1. Does a model trained directly to classify DO < 5 mg/L rank those samples better than
   thresholding the regression prediction? Reported as AUC, average precision and the
   recall available while keeping at least half of the alarms correct.
2. Does conformalized quantile regression (CQR) cover the low-DO tail better than a
   split-conformal interval of constant width? Reported overall and for DO < 5 mg/L.

Calibration uses stations carved out of the training set, never the test stations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.data import features as feat
from aquanexus.logger import get_logger
from aquanexus.ml import do_experiment as ex

log = get_logger("scripts.tail_do")


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
    parser.add_argument("--regime", default="strict")
    parser.add_argument("--level", type=float, default=0.9)
    parser.add_argument("--out", type=Path, default=Path("experiments"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.cache)
    split = next(s for s in ex.make_splits(frame) if s.name == args.regime)
    train, test = frame[split.train], frame[split.test].reset_index(drop=True)
    fit, calibration = carve_stations(train, 0.1, seed=1)
    y = test[feat.TARGET].to_numpy()
    low = y < ex.LOW_DO
    alpha = (1 - args.level) / 2

    ranking, intervals = [], []
    for name in args.sets:
        cols = feat.FEATURE_SETS[name]

        regressor = ex.fit_lgbm(fit, cols)
        pred = ex.predict_lgbm(regressor, test, cols)
        ranking.append(
            {"features": name, "scorer": "regression (-prediction)"} | ex.ranking_metrics(y, -pred)
        )
        classifier = ex.fit_low_do_classifier(fit, cols)
        prob = classifier.predict(test[cols], num_iteration=classifier.best_iteration)
        ranking.append(
            {"features": name, "scorer": "classifier"}
            | ex.ranking_metrics(y, prob)
            | ex.calibration_report(prob, y)
        )
        cal_prob = classifier.predict(calibration[cols], num_iteration=classifier.best_iteration)
        mapping = ex.isotonic_map(cal_prob, calibration[feat.TARGET].to_numpy())
        calibrated = np.interp(prob, mapping["x"], mapping["y"])
        ranking.append(
            {"features": name, "scorer": "classifier + isotonic"}
            | ex.ranking_metrics(y, calibrated)
            | ex.calibration_report(calibrated, y)
        )
        log.info("%s ranking done", name)

        lo_c, hi_c = ex.conformal_interval(regressor, calibration, test, cols, args.level)
        cal_pred = ex.predict_lgbm(regressor, calibration, cols)
        lo_b, hi_b = ex.binned_conformal_interval(
            cal_pred, calibration[feat.TARGET].to_numpy(), pred, level=args.level
        )
        lower = ex.fit_quantile(fit, cols, alpha)
        upper = ex.fit_quantile(fit, cols, 1 - alpha)
        lo_q, hi_q = ex.cqr_interval(lower, upper, calibration, test, cols, args.level)
        for kind, lo, hi in [("constant", lo_c, hi_c), ("binned", lo_b, hi_b), ("cqr", lo_q, hi_q)]:
            for group, m in [
                ("all", np.ones(len(y), bool)),
                ("do_below_5", low),
                ("do_5_to_8", (y >= 5) & (y < 8)),
            ]:  # noqa: E501
                intervals.append(
                    {"features": name, "interval": kind, "group": group, "n": int(m.sum())}
                    | ex.coverage(y[m], lo[m], hi[m])
                )
        log.info("%s intervals done", name)

    pd.DataFrame(ranking).to_csv(args.out / "do_tail_ranking.csv", index=False)
    pd.DataFrame(intervals).to_csv(args.out / "do_tail_intervals.csv", index=False)
    print(pd.DataFrame(ranking).round(3).to_string(index=False))
    print(pd.DataFrame(intervals).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
