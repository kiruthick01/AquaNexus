"""Phase 3 split conformal prediction - three coverage regimes, real data.

Split conformal's coverage guarantee is marginal and requires calibration and
test data to be exchangeable (`aquanexus.ml.uncertainty.SplitConformalModel`
docstring). This script measures what happens to that guarantee as
exchangeability is progressively broken:

A. **In-domain** - calibration and test are both random slices of the same
   Ayase distribution. Exchangeability approximately holds; this is where the
   guarantee should be closest to met.
B. **Held-out station** - calibration comes from three Ayase stations, test is
   the fourth, never touched during training or calibration (the project's
   canonical spatial-generalisation protocol, `ModelValidator`).
C. **Cross-river (Naka)** - calibration is the entire Ayase record, test is
   the independently reserved Naka holdout (`scripts/holdout_river.py`,
   `docs/HOLDOUT_RIVER.md`) - the project's existing, real distribution-shift
   benchmark (zero-shot pooled R² -0.081).

    python scripts/phase3_conformal_experiment.py
"""

from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.data.dataset import DO_FEATURES, DO_RANGE, build_water_quality_dataset
from aquanexus.data.loader import filter_stations, load_many
from aquanexus.logger import get_logger
from aquanexus.ml.uncertainty import (
    IntervalPrediction,
    SplitConformalModel,
    coverage,
    mean_width,
)

log = get_logger("scripts.phase3_conformal_experiment")

LEVEL = 0.9
CALIBRATION_FRACTION = 0.25
SEED = 42


def load_ayase() -> pd.DataFrame:
    files = sorted(glob.glob(str(settings.RAW_DIR / "waterquality" / "saitama_*.xlsx")))
    if not files:
        raise FileNotFoundError("no water quality files; run scripts/download_data.py")
    sweep_path = settings.PROCESSED_DIR / "ayase_flow_sweep.csv"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"no flow sweep at {sweep_path}; run scripts/build_geometry.py")
    observations = filter_stations(load_many(files), water_body=settings.RIVER_NAME_JA)
    return build_water_quality_dataset(observations, pd.read_csv(sweep_path))


#: Same constants as scripts/holdout_river.py - duplicated rather than
#: imported, since scripts/ is not an installed package and this project's
#: scripts are each self-contained (see e.g. train_models.py).
NAKA_WATER_BODY_JA = "中川"
NAKA_SWEEP = "naka_flow_sweep"


def load_naka() -> pd.DataFrame:
    files = sorted(glob.glob(str(settings.RAW_DIR / "waterquality" / "saitama_*.xlsx")))
    if not files:
        raise FileNotFoundError("no water quality files; run scripts/download_data.py")
    sweep_path = settings.PROCESSED_DIR / f"{NAKA_SWEEP}.csv"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"no sweep at {sweep_path}; run scripts/holdout_river.py first")
    observations = filter_stations(load_many(files), water_body=NAKA_WATER_BODY_JA)
    return build_water_quality_dataset(observations, pd.read_csv(sweep_path))


def evaluate_regime(name: str, train: pd.DataFrame, test: pd.DataFrame,
                    features: list[str], target: str) -> dict:
    test = test.reset_index(drop=True)  # so result arrays align by plain position
    model = SplitConformalModel(model_type="linear", calibration_fraction=CALIBRATION_FRACTION,
                                seed=SEED, output_range=DO_RANGE).fit(
        train[features], train[target]
    )
    result = model.predict_interval(test[features], level=LEVEL)
    y_test = test[target].to_numpy()

    summary = {
        "regime": name,
        "n_train": len(train),
        "n_calibration": model._n_cal,  # noqa: SLF001 - reporting internal split size
        "n_test": len(test),
        "coverage": coverage(y_test, result),
        "mean_width": mean_width(result),
        "mae": float(np.mean(np.abs(result.point - y_test))),
    }

    if "station" in test.columns and test["station"].nunique() > 1:
        rows = []
        for station, positions in test.groupby("station").indices.items():
            rows.append({
                "station": station, "n": len(positions),
                "coverage": coverage(y_test[positions],
                                     IntervalPrediction(result.point[positions],
                                                        result.lower[positions],
                                                        result.upper[positions], LEVEL)),
                "mean_width": float(np.mean(result.width()[positions])),
            })
        summary["per_station"] = pd.DataFrame(rows)

    return summary


def main() -> int:
    ayase = load_ayase()
    naka = load_naka()
    features = [f for f in DO_FEATURES if f in ayase.columns]
    target = "dissolved_oxygen"
    log.info("Ayase: %d rows, %d stations. Naka: %d rows, %d stations.",
             len(ayase), ayase["station"].nunique(), len(naka), naka["station"].nunique())

    results = {}

    # A. In-domain: random 80/20 train+calibration / test split, all Ayase.
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(len(ayase))
    n_test = max(10, int(round(0.2 * len(ayase))))
    test_idx, train_idx = shuffled[:n_test], shuffled[n_test:]
    in_domain_train = ayase.iloc[train_idx].reset_index(drop=True)
    in_domain_test = ayase.iloc[test_idx].reset_index(drop=True)
    results["A_in_domain"] = evaluate_regime("in_domain", in_domain_train, in_domain_test,
                                             features, target)

    # B. Held-out station (leave-one-station-out, pooled + per-station).
    rows, all_y, all_point, all_lower, all_upper = [], [], [], [], []
    for station in sorted(ayase["station"].unique()):
        train = ayase[ayase["station"] != station].reset_index(drop=True)
        test = ayase[ayase["station"] == station].reset_index(drop=True)
        model = SplitConformalModel(model_type="linear", calibration_fraction=CALIBRATION_FRACTION,
                                    seed=SEED, output_range=DO_RANGE).fit(
            train[features], train[target]
        )
        result = model.predict_interval(test[features], level=LEVEL)
        y_test = test[target].to_numpy()
        rows.append({"station": station, "n": len(test),
                     "coverage": coverage(y_test, result), "mean_width": mean_width(result)})
        all_y.append(y_test)
        all_point.append(result.point)
        all_lower.append(result.lower)
        all_upper.append(result.upper)

    pooled = IntervalPrediction(point=np.concatenate(all_point), lower=np.concatenate(all_lower),
                                upper=np.concatenate(all_upper), level=LEVEL)
    y_pooled = np.concatenate(all_y)
    results["B_held_out_station"] = {
        "regime": "held_out_station", "n_train": None, "n_calibration": None,
        "n_test": len(y_pooled), "coverage": coverage(y_pooled, pooled),
        "mean_width": mean_width(pooled),
        "mae": float(np.mean(np.abs(pooled.point - y_pooled))),
        "per_station": pd.DataFrame(rows),
    }

    # C. Cross-river (Naka), calibrated entirely on Ayase.
    results["C_cross_river_naka"] = evaluate_regime("cross_river_naka", ayase, naka,
                                                     features, target)

    print()
    print("=" * 78)
    print(f"PHASE 3 CONFORMAL PREDICTION - target {LEVEL:.0%} coverage, "
          f"calibration_fraction={CALIBRATION_FRACTION}")
    for summary in results.values():
        print(f"{summary['regime']}: n_test={summary['n_test']}  "
              f"coverage={summary['coverage']:.3f}  mean_width={summary['mean_width']:.3f}  "
              f"MAE={summary['mae']:.3f}")
        if "per_station" in summary:
            # Station names are Japanese; route through the logger, whose
            # stream is reconfigured to UTF-8 (see aquanexus.logger), rather
            # than print(), which inherits the console's cp1252 default on
            # Windows and cannot encode them.
            log.info("%s per-station:\n%s", summary["regime"],
                     summary["per_station"].round(3).to_string(index=False))
        print("-" * 78)
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
