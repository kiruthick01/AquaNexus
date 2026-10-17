"""Check that held-out stations' labels do not reach training features.

    python scripts/check_neighbor_leak.py

The default frame computes neighbour features from every station, so a training row's
``nb_do_now`` can contain a *held-out* station's reading. That never touches a test
row's own label, but it does put test-station labels into training inputs. This script
rebuilds the frame with the strict regime's held-out stations excluded from every
neighbour pool and re-scores the neighbour-based feature sets, to see whether the
published numbers survive.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aquanexus.data import features as feat
from aquanexus.logger import get_logger
from aquanexus.ml import do_experiment as ex

log = get_logger("scripts.check_neighbor_leak")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/moe"))
    parser.add_argument("--weather", type=Path, default=Path("data/cache/weather"))
    parser.add_argument("--elevation", type=Path, default=Path("data/cache/elevation.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/do_frame.parquet"))
    parser.add_argument(
        "--published", type=Path, default=Path("experiments/do_results_default.csv")
    )
    parser.add_argument("--out", type=Path, default=Path("experiments/do_neighbor_leak_check.csv"))
    args = parser.parse_args()

    cached = pd.read_parquet(args.cache, columns=["station_id", "fiscal_year"])
    held = set(
        cached.loc[
            next(s for s in ex.make_splits(cached) if s.name == "station").test, "station_id"
        ]
    )
    log.info("%d held-out stations excluded from neighbour pools", len(held))

    frame = feat.build_frame(args.raw, args.weather, args.elevation, exclude_from_neighbors=held)
    splits = {s.name: s for s in ex.make_splits(frame)}
    rows = []
    for regime in ("strict", "station"):
        train, test = frame[splits[regime].train], frame[splits[regime].test]
        for name in ("geo", "nowcast", "full"):
            cols = feat.FEATURE_SETS[name]
            booster = ex.fit_lgbm(train, cols)
            rows.append(
                {"regime": regime, "features": name, "model": "lgbm_default_no_leak"}
                | ex.metrics(test[feat.TARGET], ex.predict_lgbm(booster, test, cols))
            )
    result = pd.DataFrame(rows)
    published = pd.read_csv(args.published)
    published = published[published["model"] == "lgbm_default"]
    merged = result.merge(
        published[["regime", "features", "r2", "mae", "recall_low"]],
        on=["regime", "features"],
        suffixes=("", "_published"),
    )
    merged.to_csv(args.out, index=False)
    show = [
        "regime",
        "features",
        "r2_published",
        "r2",
        "mae_published",
        "mae",
        "recall_low_published",
        "recall_low",
    ]
    print(merged[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
