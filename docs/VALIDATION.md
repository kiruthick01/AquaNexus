# Validation Protocols

AquaNexus uses four distinct validation protocols, each answering a different
generalisation question. None substitutes for another, and mixing them - for
example fitting a scaler or selecting features on data that a later split
holds out - would invalidate whichever claim is being made. This document is
the map; each protocol's own module docstring carries the detail.

| Protocol | Question | Where |
|---|---|---|
| A. Random split | Diagnostic only - how much does an ungrouped split inflate a score? | `ml/splits.py::random_split` |
| B. Grouped / station-held-out | Does the model work on a monitoring station it has not seen? | `ml/splits.py::grouped_split`, `ml/splits.py::spatial_split`, `ml/validator.py::ModelValidator` |
| C. Temporal (date-ordered and walk-forward) | Does the model transfer to later sampling dates (`temporal_split`) / forecast forward in time (walk-forward, Phase 1) | `ml/splits.py::temporal_split`, `ml/forecasting.py::walk_forward_splits` |
| D. Cross-river / domain-shift | Does the model transfer to a river it has never seen at all? | `scripts/holdout_river.py`, `docs/HOLDOUT_RIVER.md` |

## A. Random split - diagnostic, not for use

`ml/splits.py::random_split` exists to *demonstrate* leakage, not to validate
anything. On the HSI dataset, all rows from one water-quality observation
share identical chemistry (`data/dataset.py::build_state_vectors`), so an
ungrouped split puts near-duplicates on both sides and reports a score that
is mostly memorisation. It is retained, and reported alongside the grouped
result, specifically so the gap between them is visible.

## B. Grouped / spatial - the project's canonical protocol

`ModelValidator.cross_val_predict` runs `GroupKFold` with one fold per
station - a true leave-one-station-out evaluation on the current 4-station
(Ayase) / 5-station (Naka) datasets. This is the protocol every point-model
result in `ML_METHODOLOGY.md`, and every uncertainty-method result in
`UNCERTAINTY.md`, is reported under. `spatial_split` does the equivalent for
the HSI dataset's cross-sections, holding out a contiguous block rather than
scattering it, so no held-out section sits between two training neighbours.

**Why this is the default, not random splitting:** a station identifier is
close to a row identifier at n≈100-200; a model given rows from a station in
training has evidence about that station specifically, not just about the
physical relationship the project wants to measure.

## C. Temporal

Two different claims, not one, both offered because they answer genuinely
different questions:

- **`temporal_split`** holds out the most recent sampling dates. It tests
  whether the model transfers to later dates - a real question, since
  conditions can drift over 3 years - but it is **not** a forecasting test:
  the dataset's irregular ~monthly grab sampling means there is no
  meaningful "yesterday" to forecast from (see `docs/DATA_LIMITATIONS.md`
  Phase 1). Conflating the two would misrepresent what either result claims.
- **`forecasting.walk_forward_splits`** is the actual forecasting protocol:
  expanding-window, chronological, per-station folds, with lag/lead features
  checked for real elapsed-time gaps before being trusted
  (`ml/forecasting.py`). Run against the real dataset, it currently returns
  zero usable rows at every horizon (`docs/EXPERIMENTS.md` EXP-001) - the
  protocol is implemented and tested (`tests/test_forecasting.py`, synthetic
  series), but has no real-data result yet.

Both share one hard rule, asserted in code
(`forecasting.assert_no_temporal_leakage`) rather than left to convention:
**no test-set timestamp may be less than or equal to a training-set
timestamp within the same station.** Neither protocol ever shuffles rows.

## D. Cross-river / domain-shift

`scripts/holdout_river.py` applies the Ayase model, completely unchanged, to
the independently reserved Naka river - a different catchment, different
stations, never touched during any Ayase development. This is the project's
strongest generalisation test and its headline result: pooled zero-shot R²
-0.081, with a documented split between Naka stations inside vs. outside the
Ayase model's training ranges (`docs/HOLDOUT_RIVER.md`).

Phase 3 (`docs/UNCERTAINTY.md`) reuses this exact holdout to ask the same
question of an *interval* rather than a point prediction: does conformal
coverage degrade under the same shift? The answer was the same shape as the
point-prediction result - fine on the in-range stations, and badly miscovered
(0.479 against a 0.90 target) at the one out-of-range station - which is why
a pooled coverage number is documented there as insufficient on its own.

## Rules that apply across all four protocols

- Preprocessing (imputation, scaling) is fit inside the estimator pipeline on
  the training partition only (`ml/models.py::ModelFactory`, "linear"
  branch) - never on the whole frame before splitting.
- A random split is never substituted for a grouped one to make a number
  look better; `random_split`'s only sanctioned use is the leakage
  demonstration itself.
- Every new capability added under `docs/ML_ROADMAP.md` is evaluated under
  the same protocol as the existing baseline it is compared against, so
  numbers are comparable across phases (e.g. Phase 2/3's leave-one-station-out
  evaluation matches `ModelValidator`'s exactly, not a different split that
  happens to look favourable).
