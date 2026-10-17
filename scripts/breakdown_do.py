"""Where the strict-regime DO models are good and bad: region, season, DO level.

    python scripts/breakdown_do.py

Scores the saved models from ``scripts/uncertainty_do.py`` on the strict test set
(stations unseen in training, later fiscal years), grouped by region, season and
observed DO band, beside climatology on the same rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from aquanexus.data import features as feat
from aquanexus.logger import get_logger
from aquanexus.ml import do_experiment as ex

log = get_logger("scripts.breakdown_do")

REGIONS = {
    "Hokkaido": [1],
    "Tohoku": [2, 3, 4, 5, 6, 7],
    "Kanto": [8, 9, 10, 11, 12, 13, 14],
    "Chubu": [15, 16, 17, 18, 19, 20, 21, 22, 23],
    "Kansai": [24, 25, 26, 27, 28, 29, 30],
    "Chugoku": [31, 32, 33, 34, 35],
    "Shikoku": [36, 37, 38, 39],
    "Kyushu-Okinawa": [40, 41, 42, 43, 44, 45, 46, 47],
}
REGION_OF = {p: r for r, prefs in REGIONS.items() for p in prefs}
SEASON_OF = {
    **dict.fromkeys([12, 1, 2], "winter"),
    **dict.fromkeys([3, 4, 5], "spring"),
    **dict.fromkeys([6, 7, 8], "summer"),
    **dict.fromkeys([9, 10, 11], "autumn"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", type=Path, default=Path("data/processed/do_frame.parquet"))
    parser.add_argument("--sets", nargs="*", default=["nowcast", "full"])
    parser.add_argument("--out", type=Path, default=Path("experiments/do_breakdown.csv"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.cache)
    split = next(s for s in ex.make_splits(frame) if s.name == "strict")
    train, test = frame[split.train], frame[split.test].reset_index(drop=True)
    y = test[feat.TARGET].to_numpy()
    groupings = {
        "region": test["pref_code"].map(REGION_OF).fillna("other").to_numpy(),
        "season": test["month"].map(SEASON_OF).to_numpy(),
        "do_band": pd.cut(y, [-1, 5, 8, 11, 100], labels=["<5", "5-8", "8-11", "11+"]).astype(str),
    }
    preds = {"climatology": ex.climatology(train, test)}
    for name in args.sets:
        model = lgb.Booster(model_file=f"data/models/do_strict_{name}.txt")
        preds[f"lgbm_{name}"] = ex.predict_lgbm(model, test, feat.FEATURE_SETS[name])

    rows = []
    for kind, labels in groupings.items():
        for label in sorted(set(labels)):
            m = labels == label
            for model, pred in preds.items():
                rows.append({"grouping": kind, "group": label, "model": model}
                            | ex.metrics(y[m], pred[m]))  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    index = ["grouping", "group"]
    r2 = table.pivot_table(index=index, columns="model", values="r2").add_prefix("r2_")
    mae = table.pivot_table(index=index, columns="model", values="mae").add_prefix("mae_")
    counts = table.drop_duplicates(index).set_index(index)["n"]
    print(pd.concat([counts, r2, mae], axis=1).round(2).to_string())


if __name__ == "__main__":
    main()
