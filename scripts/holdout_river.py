"""Test the Ayase model on a river it has never seen.

Everything the project claims about generalisation so far is *within* one river:
four stations on the Ayase, held out one at a time. That is a real test and a
weak one — the stations share a channel, a catchment, a sewer network and a
sampling programme.

The Naka (中川) was reserved as a held-out reach during the data audit on
2026-09-06 and never touched. It is a different river in the same prefecture,
with its own geometry, its own monitoring stations and 192 observations carrying
both discharge and dissolved oxygen — more than the Ayase's 138.

The protocol, fixed before any of it was run:

1. The Ayase model is applied **unchanged**. No refitting, no recalibration, no
   feature selection against the new river. It is loaded from disk as the API
   serves it.
2. Two references are computed *on the Naka*, because "the model transfers
   badly" and "this river is hard" are different findings:
   - **mean** — predicting the Naka's own mean, the floor;
   - **persistence** — the previous sample at the same station, which needs no
     model at all;
   - **a Naka-trained model** — the same Ridge specification fitted to the Naka
     under the same grouped cross-validation. This is the ceiling. If it also
     scores poorly, the river is the problem rather than the transfer.
3. Results are split by whether the observation lies inside the Ayase model's
   **training ranges**. The Naka carries up to 145 m3/s against the Ayase's 74,
   so a large part of it is extrapolation, and reporting one number over both
   would hide which failure is which.

Expected outcome, recorded in advance: worse than on the Ayase. The interesting
question is whether it beats the Naka's own mean.

    python scripts/holdout_river.py

Needs `data/processed/naka_flow_sweep.csv` — see `scripts/run_hecras.py`.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.data.dataset import DO_FEATURES, DO_RANGE, build_water_quality_dataset
from aquanexus.data.loader import filter_stations, load_many
from aquanexus.logger import get_logger
from aquanexus.ml.evaluator import evaluate
from aquanexus.ml.models import HabitatPredictor, ModelConfig
from aquanexus.ml.validator import ModelValidator, persistence_baseline

log = get_logger("scripts.holdout_river")

HOLDOUT_JA = "中川"
HOLDOUT_NAME = "Naka"
SWEEP = "naka_flow_sweep"


def load_holdout(water_body: str, sweep_name: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(settings.RAW_DIR / "waterquality" / "saitama_*.xlsx")))
    if not files:
        raise FileNotFoundError("no water quality files; run scripts/download_data.py")

    sweep_path = settings.PROCESSED_DIR / f"{sweep_name}.csv"
    if not sweep_path.is_file():
        raise FileNotFoundError(
            f"no sweep at {sweep_path}; build the geometry and run "
            f"scripts/run_hecras.py --water-body {water_body}"
        )

    observations = filter_stations(load_many(files), water_body=water_body)
    dataset = build_water_quality_dataset(observations, pd.read_csv(sweep_path))
    log.info("%s: %d observation(s), %d station(s)", water_body, len(dataset),
             dataset["station"].nunique())
    return dataset


def in_training_range(dataset: pd.DataFrame, manifest: dict) -> pd.Series:
    """Which rows the Ayase model has evidence for.

    A row counts as in range only if every feature it supplies lies inside the
    range that feature took in Ayase training. Anything else is extrapolation
    and is reported separately - the Naka's discharge alone reaches twice the
    Ayase's maximum.
    """
    ranges = manifest["models"]["dissolved_oxygen"]["training_ranges"]
    inside = pd.Series(True, index=dataset.index)
    for name, (low, high) in ranges.items():
        if name not in dataset.columns:
            continue
        column = dataset[name]
        inside &= column.isna() | ((column >= low) & (column <= high))
    return inside


def transfer(dataset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The Ayase model applied to the holdout river, unchanged."""
    model = HabitatPredictor.load(settings.MODELS_DIR / "dissolved_oxygen_v1.joblib")
    predictions = model.predict(dataset[features])
    return pd.Series(predictions, index=dataset.index)


def references(dataset: pd.DataFrame, features: list[str]) -> dict[str, pd.Series]:
    """The floor, the no-model baseline, and the ceiling."""
    target = dataset["dissolved_oxygen"]

    fitted_here = ModelValidator(dataset, features, "dissolved_oxygen", "station")
    native = pd.Series(
        fitted_here.cross_val_predict(
            ModelConfig(model_type="linear", output_range=DO_RANGE)),
        index=dataset.index,
    )

    return {
        "mean of this river": pd.Series(float(target.mean()), index=dataset.index),
        "persistence": persistence_baseline(dataset, "dissolved_oxygen", "station"),
        "trained on this river": native,
    }


def score(name: str, truth: pd.Series, predicted: pd.Series,
          baseline: float) -> dict:
    usable = predicted.notna() & truth.notna()
    metrics = evaluate(truth[usable], predicted[usable], name, "holdout", baseline)
    return {"model": name, "n": metrics.n, "rmse": metrics.rmse,
            "mae": metrics.mae, "r2": metrics.r2, "bias": metrics.bias}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--water-body", default=HOLDOUT_JA)
    parser.add_argument("--sweep-name", default=SWEEP)
    args = parser.parse_args()

    manifest = json.loads(
        (settings.MODELS_DIR / "manifest.json").read_text(encoding="utf-8"))
    dataset = load_holdout(args.water_body, args.sweep_name)
    features = [f for f in DO_FEATURES if f in dataset.columns]
    truth = dataset["dissolved_oxygen"]
    floor = float(truth.mean())

    transferred = transfer(dataset, features)
    rows = [score("Ayase model, unchanged", truth, transferred, floor)]
    for name, predicted in references(dataset, features).items():
        rows.append(score(name, truth, predicted, floor))
    table = pd.DataFrame(rows)

    inside = in_training_range(dataset, manifest)
    split = []
    for label, mask in (("inside Ayase training ranges", inside),
                        ("outside (extrapolation)", ~inside)):
        if mask.sum() < 2:
            continue
        split.append({
            "subset": label, "n": int(mask.sum()),
            **{k: v for k, v in score("", truth[mask], transferred[mask],
                                      floor).items() if k not in ("model", "n")},
        })

    document = report(args.water_body, dataset, table, pd.DataFrame(split), manifest)
    Path("docs/HOLDOUT_RIVER.md").write_text(document, encoding="utf-8")
    log.info("wrote docs/HOLDOUT_RIVER.md")
    print(document)
    return 0


def report(water_body: str, dataset: pd.DataFrame, table: pd.DataFrame,
           split: pd.DataFrame, manifest: dict) -> str:
    ayase = manifest["models"]["dissolved_oxygen"]["metrics"]
    transferred = table.iloc[0]
    native = table[table.model == "trained on this river"]

    lines = [
        f"# The {HOLDOUT_NAME} ({water_body}) — a river the model has never seen",
        "",
        "Generated by `scripts/holdout_river.py`.",
        "",
        "Every generalisation claim in this project until now was *within* the "
        "Ayase: four stations, held out one at a time, sharing a channel and a "
        "catchment. This applies the shipped model, unchanged, to a different "
        f"river — {len(dataset)} observations across "
        f"{dataset['station'].nunique()} stations, reserved as a holdout during "
        "the data audit and never used until now.",
        "",
        "## Results",
        "",
        "| | n | RMSE | MAE | R² | bias |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        lines.append(f"| {row.model} | {row.n} | {row.rmse:.3f} | {row.mae:.3f} | "
                     f"{row.r2:+.3f} | {row.bias:+.3f} |")

    lines += [
        "",
        f"For reference, the same model scores RMSE {ayase['rmse']:.3f} and "
        f"R² {ayase['r2']:.3f} on the Ayase under grouped cross-validation.",
    ]

    if not split.empty:
        lines += [
            "",
            "## Transfer, split by whether the model has evidence",
            "",
            "A row counts as in range only if every feature it supplies falls "
            "inside the range that feature took in Ayase training. The "
            f"{HOLDOUT_NAME} reaches 145 m³/s against the Ayase's 74, so a large "
            "part of this river is extrapolation.",
            "",
            "| subset | n | RMSE | MAE | R² | bias |",
            "|---|---|---|---|---|---|",
        ]
        for _, row in split.iterrows():
            lines.append(f"| {row.subset} | {row.n} | {row.rmse:.3f} | "
                         f"{row.mae:.3f} | {row['r2']:+.3f} | {row.bias:+.3f} |")

    # The verdict is computed, not written in advance.
    beats_floor = transferred.r2 > 0
    ceiling = float(native["r2"].iloc[0]) if len(native) else float("nan")
    lines += [
        "",
        "## Verdict",
        "",
    ]
    if beats_floor:
        lines.append(
            f"The Ayase model transfers: on a river it has never seen it scores "
            f"R² {transferred.r2:+.3f}, better than predicting that river's own "
            f"mean. "
        )
    else:
        lines.append(
            f"**The Ayase model does not transfer.** On a river it has never "
            f"seen it scores R² {transferred.r2:+.3f} — worse than predicting "
            f"that river's own mean, which needs no model at all. "
        )
    if np.isfinite(ceiling):
        lines.append(
            f"A model of the same specification fitted to this river reaches "
            f"R² {ceiling:+.3f}, so the gap between {transferred.r2:+.3f} and "
            f"{ceiling:+.3f} is what transferring costs, and the remainder is "
            f"the river being hard."
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
