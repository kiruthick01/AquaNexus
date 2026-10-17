"""Score the nationwide DO model on the Ayase (綾瀬川) and Naka (中川) stations.

    python scripts/case_study_do.py

Each river's stations are removed from training entirely, so the model has never seen
a sample from the river. Training uses fiscal years up to 2019; the test is FY2021-22
(2020 purged), as in the strict regime. Reported beside climatology (which has no
station history to use, so it falls back to prefecture-month means) and persistence.

Caveat for ``nowcast`` and ``full``: their neighbour features read other stations
within 25 km, which can include other stations on the same river. That is the real
deployment situation for a monitored river, but it is not "an unseen river"; the
``geo`` rows (no neighbour or history features) are the unseen-river result.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aquanexus.data import features as feat
from aquanexus.data import moe
from aquanexus.logger import get_logger
from aquanexus.ml import do_experiment as ex

log = get_logger("scripts.case_study_do")

RIVERS = {"Ayase": ("綾瀬川", {11, 13}), "Naka": ("中川", {11, 12, 13})}


def river_stations(raw: Path, name: str, prefectures: set[int]) -> set:
    stations = moe.load_stations(moe.find_files(raw, "mm"))
    hit = stations["water_body"].astype(str).str.contains(name)
    pref = stations["station_id"].astype(str).str[:-5].astype(int)
    return set(stations.loc[hit & pref.isin(prefectures), "station_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/moe"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/do_frame.parquet"))
    parser.add_argument("--out", type=Path, default=Path("experiments/do_case_studies.csv"))
    args = parser.parse_args()

    frame = pd.read_parquet(args.cache)
    rows = []
    for river, (name, prefs) in RIVERS.items():
        ids = river_stations(args.raw, name, prefs)
        held = frame["station_id"].isin(ids)
        train_mask = (frame["fiscal_year"] <= 2019) & ~held
        test_mask = (frame["fiscal_year"] >= 2021) & held
        train, test = frame[train_mask], frame[test_mask]
        log.info("%s: %d stations, train %d, test %d", river, len(ids), len(train), len(test))

        def record(model, features, pred, river=river, test=test, ids=ids):
            rows.append(
                {"river": river, "stations": len(ids), "model": model, "features": features}
                | ex.metrics(test[feat.TARGET], pred)
            )

        record("climatology", "-", ex.climatology(train, test))
        record("persistence", "history", ex.persistence(test))
        for fs in ("geo", "geo_lu", "nowcast", "nowcast_lu", "full"):
            cols = feat.FEATURE_SETS[fs]
            booster = ex.fit_lgbm(train, cols)
            record("lgbm_default", fs, ex.predict_lgbm(booster, test, cols))

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    show = ["river", "stations", "model", "features", "n", "r2", "mae", "rmse", "mae_low"]
    print(table[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
