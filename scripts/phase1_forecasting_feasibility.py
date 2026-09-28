"""Phase 1 forecasting feasibility check - run against the real dataset.

This does not train a forecasting model. It runs the temporal infrastructure in
``aquanexus.ml.forecasting`` against the actual canonical dissolved-oxygen record
and counts how many rows would survive as usable (lag-complete, leakage-free,
within-tolerance) forecasting samples at each horizon. See
``docs/ML_ROADMAP.md`` Phase 1 and ``docs/DATA_LIMITATIONS.md`` for the reasoning
behind the tolerance used.

If, as expected from the Phase 0 audit, the usable count is too small to report a
train/test metric, this script says so explicitly and does not fit anything.

    python scripts/phase1_forecasting_feasibility.py
"""

from __future__ import annotations

import glob
import sys

import pandas as pd

from aquanexus.config import settings
from aquanexus.data.dataset import build_water_quality_dataset
from aquanexus.data.loader import filter_stations, load_many
from aquanexus.logger import get_logger
from aquanexus.ml.forecasting import add_lag_features, add_lead_targets, walk_forward_splits

log = get_logger("scripts.phase1_forecasting_feasibility")

LAGS = (1, 3, 7)
HORIZONS = (1, 3, 7)
MIN_TEST_POINTS_PER_FOLD = 10
MIN_FOLDS = 5


def usable_counts(dataset: pd.DataFrame) -> pd.DataFrame:
    """Rows with a complete set of lag features AND a lead target, per horizon."""
    tagged = add_lag_features(dataset, "dissolved_oxygen", lags=LAGS)
    rows = []
    for horizon in HORIZONS:
        with_target = add_lead_targets(tagged, "dissolved_oxygen", horizons=(horizon,))
        lag_cols = [f"dissolved_oxygen_lag{lag:g}d" for lag in LAGS]
        target_col = f"dissolved_oxygen_lead{horizon:g}d"
        complete = with_target.dropna(subset=[*lag_cols, target_col])
        per_station = complete.groupby("station").size()
        rows.append({
            "horizon_days": horizon,
            "usable_rows_total": len(complete),
            "usable_rows_by_station": per_station.to_dict(),
        })
    return pd.DataFrame(rows)


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
    log.info("canonical dataset: %d rows, %d stations", len(dataset),
             dataset["station"].nunique())

    counts = usable_counts(dataset)
    log.info("usable (lag-complete, within-tolerance) forecasting rows by horizon:\n%s",
             counts.to_string(index=False))

    folds = walk_forward_splits(dataset, n_folds=MIN_FOLDS)
    log.info("walk-forward folds obtainable at n_folds=%d: %d (0 means no station has "
             "enough distinct sampling dates)", MIN_FOLDS, len(folds))

    max_usable = int(counts["usable_rows_total"].max()) if len(counts) else 0
    viable = max_usable >= MIN_FOLDS * MIN_TEST_POINTS_PER_FOLD and len(folds) >= MIN_FOLDS

    print()
    print("=" * 70)
    if viable:
        print("UNEXPECTED: usable sample count clears the walk-forward minimum.")
        print("Re-check docs/ML_ROADMAP.md Phase 1 status before proceeding to modeling.")
    else:
        print("PHASE 1 FORECASTING: BLOCKED BY DATA AVAILABILITY")
        print(f"Best horizon usable rows: {max_usable} "
              f"(need >= {MIN_FOLDS * MIN_TEST_POINTS_PER_FOLD} for "
              f"{MIN_FOLDS} walk-forward folds of {MIN_TEST_POINTS_PER_FOLD} test points each)")
        print(f"Walk-forward folds obtainable: {len(folds)} (need >= {MIN_FOLDS})")
        print("No forecasting model was fit. No MAE/RMSE/R2 is reported for this phase.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
