"""Phase 6 transfer learning - Ayase (source) to Naka (target), real data.

The existing cross-river holdout (`scripts/holdout_river.py`) already measures
zero-shot transfer on the full 192-row Naka record: pooled R2 -0.081
(`docs/HOLDOUT_RIVER.md`). This script does not recompute that number - it
cites it - and instead evaluates three methods that actually use some
target-domain information, on a held-out Naka test set that stays disjoint
from whatever target information each method uses:

- domain-aligned prediction: target feature statistics, no target labels
- fine-tuned Ridge: a small labelled target slice, shrunk toward the source
  model's own coefficients (not an independent retrain)
- frozen-feature-extractor + target head: the same small slice, adapting
  only the final layer of the Phase 5 MLP

A random 15% row-level slice of the real Naka dataset is the "small held-in
target slice" every adaptation method may use; the remaining 85% is the
genuine, never-touched test set every method - including zero-shot,
recomputed on this exact subset for a fair comparison - is scored on.

    python scripts/phase6_transfer_learning_experiment.py
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
from aquanexus.ml.deep import MLPModel
from aquanexus.ml.evaluator import evaluate
from aquanexus.ml.models import HabitatPredictor, ModelConfig
from aquanexus.ml.transfer import (
    FrozenHeadAdapter,
    domain_aligned_predict,
    fine_tune_ridge,
    predict_fine_tuned,
)

log = get_logger("scripts.phase6_transfer_learning_experiment")

SEED = 42
ADAPTATION_FRACTION = 0.15
#: Swept rather than fixed at one value: an initial single-value attempt
#: (shrinkage=10) failed catastrophically (see docs/TRANSFER_LEARNING.md), and
#: understanding why required seeing the whole curve, not picking whichever
#: single value scores best on the held-out set - that would be tuning
#: against the evaluation data, which is exactly the leakage this project's
#: validation rules forbid.
FINE_TUNE_SHRINKAGES = (1.0, 10.0, 100.0, 1000.0, 10000.0)


def load_river(water_body: str, sweep_name: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(settings.RAW_DIR / "waterquality" / "saitama_*.xlsx")))
    if not files:
        raise FileNotFoundError("no water quality files; run scripts/download_data.py")
    sweep_path = settings.PROCESSED_DIR / f"{sweep_name}.csv"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"no sweep at {sweep_path}")
    observations = filter_stations(load_many(files), water_body=water_body)
    return build_water_quality_dataset(observations, pd.read_csv(sweep_path))


def main() -> int:
    ayase = load_river(settings.RIVER_NAME_JA, "ayase_flow_sweep")
    naka = load_river("中川", "naka_flow_sweep")
    features = [f for f in DO_FEATURES if f in ayase.columns]
    target = "dissolved_oxygen"
    log.info("Ayase (source): %d rows, %d stations. Naka (target): %d rows, %d stations.",
             len(ayase), ayase["station"].nunique(), len(naka), naka["station"].nunique())

    source_ridge = HabitatPredictor(ModelConfig(model_type="linear", output_range=DO_RANGE)
                                    ).fit(ayase[features], ayase[target])
    source_mlp = MLPModel(hidden_sizes=(16, 8), max_epochs=300, patience=20, seed=SEED,
                          output_range=DO_RANGE).fit(ayase[features], ayase[target])

    rng = np.random.default_rng(SEED)
    n = len(naka)
    shuffled = rng.permutation(n)
    n_adapt = max(10, int(round(n * ADAPTATION_FRACTION)))
    adapt_idx, test_idx = shuffled[:n_adapt], shuffled[n_adapt:]
    naka_adapt = naka.iloc[adapt_idx].reset_index(drop=True)
    naka_test = naka.iloc[test_idx].reset_index(drop=True)
    log.info("Naka split: %d adaptation row(s), %d genuinely held-out test row(s)",
             len(naka_adapt), len(naka_test))

    y_test = naka_test[target].to_numpy()
    baseline = float(y_test.mean())
    rows = []

    zero_shot_pred = source_ridge.predict(naka_test[features])
    rows.append(evaluate(y_test, zero_shot_pred, "zero_shot (Ridge, unchanged)",
                         "naka-held-out-subset", baseline))

    aligned_pred = domain_aligned_predict(source_ridge, naka_test[features])
    rows.append(evaluate(y_test, aligned_pred, "domain_aligned (no target labels)",
                         "naka-held-out-subset", baseline))

    for shrinkage in FINE_TUNE_SHRINKAGES:
        coefficients, intercept = fine_tune_ridge(source_ridge, naka_adapt[features],
                                                  naka_adapt[target], shrinkage=shrinkage)
        fine_tuned_pred = predict_fine_tuned(source_ridge, naka_test[features],
                                            coefficients, intercept)
        rows.append(evaluate(y_test, fine_tuned_pred, f"fine_tuned (shrinkage={shrinkage:g})",
                             "naka-held-out-subset", baseline))

    frozen_head = FrozenHeadAdapter(source_mlp, max_epochs=200, patience=20, seed=SEED).fit(
        naka_adapt[features], naka_adapt[target]
    )
    frozen_head_pred = frozen_head.predict(naka_test[features])
    rows.append(evaluate(y_test, frozen_head_pred, "frozen_head (MLP)",
                         "naka-held-out-subset", baseline))

    table = pd.DataFrame([{"model": m.model, "n": m.n, "rmse": m.rmse, "mae": m.mae,
                          "r2": m.r2, "bias": m.bias} for m in rows])

    print()
    print("=" * 78)
    print(f"PHASE 6 TRANSFER LEARNING - Naka held-out test subset (n={len(naka_test)})")
    print(table.round(3).to_string(index=False))
    print("-" * 78)
    print("For reference (not recomputed here): zero-shot on the FULL 192-row Naka "
          "record scores pooled R2 -0.081 (docs/HOLDOUT_RIVER.md). The zero-shot row "
          "above is the same unchanged model, scored only on this run's held-out subset.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
