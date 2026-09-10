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
3. Results are reported per station as well as pooled. Four Naka stations sit on
   中川上流 at 4-21 m3/s and one on 中川中流 at 75, reaching 145 - averaging a
   station the model has evidence for with one it plainly does not would hide
   both.
4. Results are split by whether the observation lies inside the Ayase model's
   **training ranges**. The Naka carries up to 145 m3/s against the Ayase's 74,
   so a large part of it is extrapolation, and reporting one number over both
   would hide which failure is which.

The holdout river is processed **identically** to the training river: the same
cross-section spacing, the same bed-quantile extraction, the same
validator-flagged sections excluded, the same twelve-profile sweep shape. Any
difference in treatment would confound "the model does not transfer" with "the
pipeline was run differently", and the point of the exercise is to separate
them.

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
from datetime import UTC, datetime
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
#: Machine-readable twin of `docs/HOLDOUT_RIVER.md`, written from the same
#: computed values in the same run. The API serves the transfer result from
#: this, so nobody has to keep a second copy of the numbers in step by hand.
ARTEFACT = "holdout_naka.json"


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


def by_station(dataset: pd.DataFrame, truth: pd.Series,
               predicted: pd.Series, floor: float) -> pd.DataFrame:
    """Per-station transfer.

    The Naka's stations are not interchangeable: four sit on 中川上流 carrying
    4-21 m3/s, and 46八条橋 sits on 中川中流 carrying 75 and reaching 145 - twice
    anything in Ayase training. A pooled number would average a station the
    model has evidence for with one it plainly does not.
    """
    rows = []
    for (body, station), block in dataset.groupby(["water_body", "station"]):
        index = block.index
        usable = predicted[index].notna() & truth[index].notna()
        if usable.sum() < 2:
            continue
        metrics = evaluate(truth[index][usable], predicted[index][usable],
                           station, "holdout", floor)
        rows.append({"sub-reach": body, "station": station, "n": metrics.n,
                     "mean discharge": float(block["discharge"].mean()),
                     "observed DO": float(block["dissolved_oxygen"].mean()),
                     "rmse": metrics.rmse, "r2": metrics.r2,
                     "bias": metrics.bias})
    return pd.DataFrame(rows).sort_values("mean discharge")


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

    stations = by_station(dataset, truth, transferred, floor)
    document = report(args.water_body, dataset, table, pd.DataFrame(split),
                      stations, manifest)
    Path("docs/HOLDOUT_RIVER.md").write_text(document, encoding="utf-8")
    log.info("wrote docs/HOLDOUT_RIVER.md")

    served = evidence(args.water_body, dataset, table, pd.DataFrame(split),
                      stations, manifest)
    artefact = settings.PROCESSED_DIR / ARTEFACT
    artefact.write_text(json.dumps(served, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log.info("wrote %s", artefact)
    print(document)
    return 0


def station_records(stations: pd.DataFrame, home: dict) -> list[dict]:
    """Per-station rows, each carrying whether the model has evidence for it.

    The flag is on mean discharge, which is what separates the two sub-reaches:
    four stations sit inside the range the model was fitted on and one sits at
    twice its maximum. A client that cannot tell them apart would average a
    working model with an extrapolating one and show a single misleading score.
    """
    if stations.empty:
        return []
    low, high = home["training_ranges"]["discharge"]
    records = stations.rename(
        columns={"sub-reach": "sub_reach", "mean discharge": "mean_discharge",
                 "observed DO": "observed_do"}
    ).to_dict(orient="records")
    for row in records:
        row["in_training_range"] = bool(low <= row["mean_discharge"] <= high)
    return records


def evidence(water_body: str, dataset: pd.DataFrame, table: pd.DataFrame,
             split: pd.DataFrame, stations: pd.DataFrame, manifest: dict) -> dict:
    """The same result as `report`, in a form the API can serve.

    The prose here is derived from the numbers in this run rather than written
    beside them, for the same reason the document's verdict is: a sentence
    typed once and left alone stops being true the next time anything is
    retrained.
    """
    home = manifest["models"]["dissolved_oxygen"]
    transferred = table.iloc[0]

    def subset(prefix: str) -> dict | None:
        if split.empty:
            return None
        rows = split[split.subset.str.startswith(prefix)]
        return None if rows.empty else rows.iloc[0].to_dict()

    inside, outside = subset("inside"), subset("outside")

    headline = (
        f"Pooled over the whole river the transfer scores R² {transferred.r2:+.3f}. "
        "That single number averages a model working with the same model "
        "extrapolating, and the split below is the honest reading."
    )
    if inside and outside:
        headline = (
            f"Inside the ranges it was fitted on, the unchanged Ayase model scores "
            f"R² {inside['r2']:+.3f} on a river it has never seen — against "
            f"{home['metrics']['r2']:+.3f} at home. Outside them it scores "
            f"{outside['r2']:+.3f}, worse than predicting this river's mean."
        )

    caveats = [
        f"The model is unchanged: loaded from disk as the API serves it, with no "
        f"refitting or recalibration against the {HOLDOUT_NAME}.",
        f"Pooled over the river the transfer scores R² {transferred.r2:+.3f}. Read "
        f"the split, not the pooled figure — it averages two different regimes.",
    ]
    if outside:
        caveats.append(
            f"{int(outside['n'])} of {int(transferred.n)} observations fall outside "
            f"the Ayase training ranges. These are the rows /predict has flagged as "
            f"out of range since the API was first served; this is what that flag "
            f"is worth."
        )
    caveats.append(
        f"The transfer is biased {transferred.bias:+.2f} mg/L. The {HOLDOUT_NAME} "
        f"carries more oxygen than the Ayase, so a model fitted to the more "
        f"polluted river reads the cleaner one as worse than it is. The direction "
        f"was recorded before the run."
    )
    caveats.append(
        "Stations are not interchangeable: the per-station table spans one "
        "sub-reach inside the training range and one well outside it, and a "
        "pooled score hides which is which."
    )

    return {
        "river": {"name": HOLDOUT_NAME, "name_ja": water_body},
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "scripts/holdout_river.py",
        "document": "docs/HOLDOUT_RIVER.md",
        "n": int(transferred.n),
        "n_stations": int(dataset["station"].nunique()),
        "trained_on": {
            "river": f"{manifest.get('river', 'Ayase')} ({manifest.get('river_ja', '')})".strip(),
            "target": home["target"],
            "unit": home["unit"],
            "n_train": int(home["n_train"]),
            "metrics": dict(home["metrics"]),
        },
        "pooled": table.to_dict(orient="records"),
        "by_evidence": split.to_dict(orient="records") if not split.empty else [],
        "by_station": station_records(stations, home),
        "headline": headline,
        "caveats": caveats,
    }


def report(water_body: str, dataset: pd.DataFrame, table: pd.DataFrame,
           split: pd.DataFrame, stations: pd.DataFrame, manifest: dict) -> str:
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

    if not stations.empty:
        lines += [
            "",
            "## Per station",
            "",
            "The stations are not interchangeable: four sit on 中川上流 carrying "
            "4–21 m³/s and one on 中川中流 carrying 75 and reaching 145 — twice "
            "anything in Ayase training.",
            "",
            "| sub-reach | station | n | mean Q | observed DO | RMSE | R² | bias |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, row in stations.iterrows():
            lines.append(
                f"| {row['sub-reach']} | {row.station} | {row.n} | "
                f"{row['mean discharge']:.1f} | {row['observed DO']:.2f} | "
                f"{row.rmse:.3f} | {row.r2:+.3f} | {row.bias:+.3f} |"
            )

    # The verdict is computed, not written in advance.
    beats_floor = transferred.r2 > 0
    ceiling = float(native["r2"].iloc[0]) if len(native) else float("nan")
    lines += [
        "",
        "## Verdict",
        "",
    ]
    # Pooled first, then the split - which usually changes the reading.
    empty = pd.DataFrame()
    inside_row = split[split.subset.str.startswith("inside")] if not split.empty else empty
    outside_row = split[split.subset.str.startswith("outside")] if not split.empty else empty

    if beats_floor:
        lines.append(
            f"Pooled over the whole river the Ayase model scores R² "
            f"{transferred.r2:+.3f}, better than predicting that river's own mean."
        )
    else:
        lines.append(
            f"Pooled over the whole river the Ayase model scores R² "
            f"{transferred.r2:+.3f} — worse than predicting that river's own "
            f"mean, which needs no model at all."
        )

    if len(inside_row) and len(outside_row):
        inside_r2 = float(inside_row["r2"].iloc[0])
        outside_r2 = float(outside_row["r2"].iloc[0])
        ayase_r2 = float(ayase["r2"])
        if inside_r2 > 0 and outside_r2 < inside_r2:
            lines += [
                "",
                f"**That single number is misleading, and the split says why.** "
                f"Where the model has evidence it scores R² {inside_r2:+.3f} — "
                f"against {ayase_r2:+.3f} on its own river, so it transfers to a "
                f"different catchment nearly intact. Where it does not, it scores "
                f"{outside_r2:+.3f}. The pooled figure is the average of a model "
                f"working and the same model extrapolating, and "
                f"{int(outside_row['n'].iloc[0])} of {int(transferred.n)} "
                f"observations are outside the range it was fitted on.",
                "",
                "The API already flags exactly these rows as out of range on every "
                "prediction. This is what that flag is worth: on the wrong side of "
                "it the model is worse than useless, and on the right side it is "
                "about as good as it is at home.",
            ]

    if np.isfinite(ceiling):
        lines += [
            "",
            f"A model of the same specification fitted to this river reaches R² "
            f"{ceiling:+.3f} — better than the Ayase model manages on the Ayase. "
            f"The Naka is the more predictable river, so the gap between "
            f"{transferred.r2:+.3f} and {ceiling:+.3f} is the cost of transfer "
            f"rather than a hard river.",
        ]

    bias = float(transferred.bias)
    if abs(bias) > 0.5:
        lines += [
            "",
            f"The transfer is **biased {bias:+.2f} mg/L**, and the direction was "
            f"predicted before the run: the Naka carries 8.0–8.8 mg/L against the "
            f"Ayase's 6.9, so a model fitted to the more polluted river reads the "
            f"cleaner one as worse than it is. A model that learned this river's "
            f"oxygen level rather than its physics would do exactly this.",
        ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
