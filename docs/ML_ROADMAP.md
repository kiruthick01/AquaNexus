# ML Roadmap — Phase 0 Repository Audit

This document is the output of a read-only audit (no code changed). It records what
AquaNexus already does, what it does not, and — for each of the eight capabilities
requested (forecasting, uncertainty, conformal prediction, remote sensing, deep
learning, transfer learning, Bayesian modeling, GNNs) — whether the current data
actually supports it, with concrete numbers rather than assumption.

Companion document: `docs/ML_METHODOLOGY.md` already contains most of the underlying
findings this roadmap builds on (splits, benchmarks, ablations, held-out-river result).
This file does not repeat those results; it cites them and asks what is still missing.

---

## 1. Current capabilities (with file references)

| Capability | Status | Where |
|---|---|---|
| Data ingestion (water quality) | Done | `src/aquanexus/data/loader.py` — `load_water_quality`, `discover_parameters`, `apply_censoring` reads `data/raw/waterquality/saitama_{2022,2023,2024}.xlsx` |
| Data ingestion (HEC-RAS hydraulics) | Done | `src/aquanexus/hecras/reader.py`, `project.py`, `runner.py` |
| Data ingestion (bathymetry / point cloud) | Done | `src/aquanexus/data/pointcloud.py`, `src/aquanexus/hecras/geometry.py` (`extract_cross_section`, `river_station` ordering) |
| Preprocessing / derived hydraulics | Done | `src/aquanexus/data/preprocessor.py` — `do_saturation`, `froude_number`, `bed_shear_stress`, `add_derived_features` |
| Feature engineering (join sim↔observed) | Done | `src/aquanexus/data/dataset.py` — `interpolate_hydraulics`, `build_water_quality_dataset`, `feature_columns` |
| Synthetic label generation (HSI) | Done, clearly flagged synthetic | `src/aquanexus/data/synthetic.py` — `habitat_suitability`, `falsification_report` |
| Data validation | Done | `src/aquanexus/data/validator.py` — `validate_sections`, `validate_observations` |
| Classical ML models | Done | `src/aquanexus/ml/models.py` — `ModelFactory` (Ridge, RandomForest, XGBoost, MeanPredictor) |
| Training / model selection | Done | `src/aquanexus/ml/trainer.py` — `benchmark`, `select_best` |
| Evaluation metrics | Done | `src/aquanexus/ml/evaluator.py` — `Metrics`, `compare`, `residual_summary` (MAE/RMSE/R²/bias — no misapplied "accuracy") |
| Random split (diagnostic only) | Done | `src/aquanexus/ml/splits.py::random_split` — explicitly documented as leaky, kept only to demonstrate the leakage gap |
| Grouped split (observation-held-out) | Done | `src/aquanexus/ml/splits.py::grouped_split` |
| Spatial split (station/cross-section held out) | Done | `src/aquanexus/ml/splits.py::spatial_split` |
| Flow-extremes split (reframed event-based) | Done | `src/aquanexus/ml/splits.py::flow_split` |
| Date-ordered split (weaker "temporal" claim) | Done, explicitly *not* forecasting | `src/aquanexus/ml/splits.py::temporal_split` — docstring states this tests transfer to later sampling dates, not chronology |
| Cross-river / domain-shift validation | Done, is the project's headline result | `src/aquanexus/api/routes/holdout.py`, `data/processed/holdout_naka.json`, `docs/HOLDOUT_RIVER.md` |
| SHAP explainability | Done | `src/aquanexus/ml/explainer.py` — `HabitatExplainer`, `collinear_pairs`, dispatches TreeExplainer vs linear explainer by model type |
| Scenario analysis | Done | `src/aquanexus/api/routes/scenarios.py` |
| FastAPI | Done | `src/aquanexus/api/app.py`, `routes/{predictions,scenarios,holdout,explanations,health}.py` |
| Frontend/dashboard | Done | `frontend/src/{pages,components,services}`, `src/aquanexus/dashboard/` (Streamlit) |
| Automated tests | Done, 378 test functions across 18 files in `tests/` | matches memory's "388 backend" figure closely enough (recount here found 378; treat either as current, both far exceed a smoke suite) |
| Synthetic-vs-real label provenance | Done, actively enforced | `docs/ML_METHODOLOGY.md` "Standing caveat" + `falsification_report`; two models served (`dissolved_oxygen` = real labels, n=138, R² 0.394; `hsi` = synthetic, R² 0.99 = function recovery, never presented as the headline) |
| Config / dependency management | Done | `pyproject.toml` — extras split into `ml`, `geo`, `api`, `dashboard`, `hecras`, `dev` |

**Canonical dataset:** `data/processed/ayase_do_dataset.csv`, built by
`aquanexus.data.dataset.build_water_quality_dataset()`. 138 rows, 4 stations, 15 columns
(`discharge, dissolved_oxygen, reach_depth, reach_velocity, reach_top_width,
reach_froude, water_temp, air_temp, station, water_body, timestamp, month, month_sin,
month_cos, do_saturation`). This is the *real-label* dataset; the separate 7,314-row
HSI dataset exists for the synthetic-label benchmark and is a different canonical
frame for a different, clearly-flagged, purpose.

**Canonical evaluation protocol:** grouped cross-validation with each of the 4
stations held out in turn (`aquanexus.ml.validator.ModelValidator`, results in
`docs/ML_METHODOLOGY.md` "Phase 2c"). Ridge currently wins (RMSE 1.785, R² 0.394),
beating persistence only marginally (R² 0.385) and losing to it on MAE. This is the
baseline every new technique in this roadmap must be compared against — not the
synthetic-HSI R² 0.99, which measures function recovery, not skill.

---

## 2. Duplicated functionality / technical debt found

- `ML_STRATEGY.md` (repo root) is the original plan and is **known stale** — it
  assumes a 4-year hourly series that was never built. `docs/ML_METHODOLOGY.md` and
  `src/aquanexus/ml/splits.py` both explicitly document each place they depart from
  it. Not a bug, but a real risk for anyone who reads `ML_STRATEGY.md` first: it must
  stay clearly marked superseded, not deleted (it documents the correction itself).
- `build/lib/aquanexus/` is a stale build artifact duplicating `src/aquanexus/` — not
  imported by anything live, but present in the tree and could be mistaken for a
  second source of truth. Housekeeping, not touched in this audit.
- No lat/lon coordinates for the 4 (Ayase) / 5 (Naka) water-quality stations exist
  anywhere in code — `pointcloud.py` has lon/lat helpers for bathymetry *tiles*, and
  `ARCHITECTURE.md` fixes `EPSG:6677` for the point-cloud geometry, but the
  monitoring-station identifiers (`52内匠橋`, `55畷橋`, `54槐戸橋`,
  `57綾瀬川合流点前`) are names only. This blocks Phase 4 directly (§6 below).
- No explicit upstream/downstream topology between the 4 monitoring stations is
  encoded. `river_station` orders HEC-RAS *cross-sections* within one reach (spatial
  interpolation grid), not the monitoring stations themselves. This blocks Phase 8
  (§6 below) independent of the node-count problem.

No functional duplication (two competing implementations of the same thing) was
found in `src/aquanexus/`.

---

## 3. The load-bearing fact for this whole roadmap

`src/aquanexus/ml/splits.py` (module docstring) and `docs/ML_METHODOLOGY.md` already
establish, in the existing codebase, that **the dataset is a designed experiment over
flow space, not a chronology**. Verified directly against the current data rather than
taken on the docstring's word:

- 138 real DO observations, 4 stations, timestamps span 2022-04-07 to 2025-03-07
  (`ayase_do_dataset.csv`).
- Per station: 52内匠橋 n=48 (36 unique dates), 54槐戸橋 n=36 (36 dates), 55畷橋
  n=36 (36 dates), 57綾瀬川合流点前 n=18 (18 dates) over the same ~3-year span.
- That is irregular grab sampling roughly monthly per station, with gaps of weeks to
  months — not a continuous or even fixed-interval series.
- `temporal_split()` exists and is honest about what it tests: transfer to later
  *sampling dates*, not forecasting. No lag/rolling features are computed anywhere
  in `preprocessor.py`, `dataset.py`, or `synthetic.py` — deliberately, per
  `ML_METHODOLOGY.md` §"Corrections to ML_STRATEGY.md §3.1".

This single fact is why Phase 1 (forecasting) and the LSTM half of Phase 5 are
**blocked by data availability** below, not a matter of effort.

---

## 4. Missing capabilities — status and requirements

### Phase 1 — Time-series forecasting: **IMPLEMENTED (infrastructure); BLOCKED (result)**

Requirement to unblock, derived (not assumed): walk-forward validation needs multiple
non-overlapping chronological folds per station to say anything about generalization
across time. With 4 stations and the current 18–48 irregular samples each, even a
single train/test split per station leaves single-digit test points — not enough to
report an MAE/RMSE with a meaningful confidence interval, let alone 3+ walk-forward
folds. A defensible minimum: **at least ~50–100 observations per station at a fixed,
regular sampling interval** (e.g. daily or weekly, not ad hoc monthly grabs), spanning
at least 2 full seasonal cycles, so a 5-fold walk-forward split leaves ≥10 test points
per fold. None of the three ingested years meet this; the real record is grab-sample
monitoring, not sensor logging.

**Status update (post-implementation):** the infrastructure predicted above was built
in `src/aquanexus/ml/forecasting.py` — density-aware lag features, lead targets
(forecast labels), time-based rolling stats, a walk-forward chronological splitter, and
a leakage-detection assertion — tested against synthetic series only
(`tests/test_forecasting.py`, 14 tests). Run against the real canonical dataset via
`scripts/phase1_forecasting_feasibility.py`, the blocker is now a measured fact, not a
projection: **0 of 138 rows are usable (lag-complete and lead-target within tolerance)
at any of DO(t+1), DO(t+3), DO(t+7)**. See `docs/EXPERIMENTS.md` EXP-001 and
`docs/DATA_LIMITATIONS.md`. No forecasting model was fit; no MAE/RMSE/R² is reported
for this phase. Implemented as a single flat module (`ml/forecasting.py`), not the
`ml/forecasting/` subpackage sketched in §5 below — the existing `ml/` package is flat
(`splits.py`, `evaluator.py`, etc.), and a one-file module matches that convention more
closely than a new subpackage would.

### Phase 2 — Predictive uncertainty (bootstrap/ensemble, quantile regression): **IMPLEMENTED**

Implemented in `src/aquanexus/ml/uncertainty.py` (`BootstrapIntervalModel`,
`QuantileIntervalModel`, both behind a shared `IntervalModel` interface), tested in
`tests/test_uncertainty.py`, and run against the real dataset via
`scripts/phase2_uncertainty_experiment.py`. Confirmed, not merely predicted: bootstrap
achieves near-nominal 90% coverage (0.920 pooled) but wide intervals (mean 6.74 mg/L);
quantile regression is sharper (3.41 mg/L) but undercovers badly (0.703 pooled, as low
as 0.521 at one station) and should not be treated as calibrated at this sample size.
Full results and per-station breakdown in `docs/UNCERTAINTY.md` and
`docs/EXPERIMENTS.md` EXP-002.

### Phase 3 — Conformal prediction: **VIABLE, with a tight-calibration-set caveat**

Split conformal needs training/calibration/test to be disjoint. With only 4 Ayase
stations, holding out a whole station for calibration leaves 3 for training — usable,
but calibration-set size will be small (18–48 points) and coverage estimates will
carry wide uncertainty of their own; this must be stated, not smoothed over. The
existing cross-river holdout (`holdout_naka.json`, n=192, 5 stations) is directly
usable as the distribution-shift coverage check this phase requires — infrastructure
for that comparison already exists via the `/holdout` route.

### Phase 4 — Satellite / remote sensing: **BLOCKED by missing station coordinates**

The observation dates (2022–2025) are well within Sentinel-2 (2015–) and Landsat
coverage, so temporal compatibility is not the blocker. The blocker is that **no
lat/lon exists for the 4 (or 5 Naka) monitoring stations anywhere in this repo** —
only station names. `geopandas`/`rasterio` are already declared as optional deps
(`pyproject.toml` `geo` extra), but no satellite API client
(e.g. `pystac`/`sentinelsat`/`planetary-computer`) is present. Before any pixel is
fetched, the station coordinates must be sourced from a citable public record (e.g.
the Saitama monitoring program's own station registry) — not estimated from station
names — and that provenance documented. This is a data-acquisition prerequisite, not
an implementation task, and should not be worked around by guessing coordinates.

### Phase 5 — Deep learning (MLP / LSTM): **PARTIALLY BLOCKED**

MLP baseline is technically runnable on 138 rows but should be expected to *lose* to
Ridge, consistent with the existing finding that XGBoost already overfits at this
sample size (`ML_METHODOLOGY.md` Phase 2a/2c: linear beats both tree models). Framing
an MLP result as competitive without that caveat would misrepresent the dataset. LSTM
requires the same chronological density Phase 1 lacks — blocked for the identical
reason, not independently.

### Phase 6 — Transfer learning / domain adaptation: **VIABLE, infrastructure already exists**

Ayase (source, n=138, 4 stations) → Naka (target, n=192, 5 stations) is already
measured as a **zero-shot** cross-river transfer: pooled R² −0.081, in-training-range
R² +0.336 (vs +0.394 at home), out-of-range R² −0.903 (`HOLDOUT_RIVER.md`). No model
has ever been fit on Naka data on purpose. What is genuinely missing is everything
past zero-shot: fine-tuning on a small held-in Naka slice, a frozen-feature-extractor
+ target-head variant, or explicit feature-distribution alignment — and clearly
labeling each as what it is, since "zero-shot transfer" (the existing result) is not
"fine-tuning" and must not be conflated with it going forward.

### Phase 7 — Bayesian modeling: **VIABLE**

n=138 with 4 station groups is a reasonable, small case for Bayesian linear
regression and a hierarchical (partial-pooling-by-station) variant — arguably a
better fit for 4 groups than a classical fixed-effect model. No Bayesian library is
currently a dependency (`pymc`/`numpyro`/`stan` all absent from `pyproject.toml`);
one will need to be added and justified as a new optional extra.

### Phase 8 — Graph Neural Network: **BLOCKED by network size and missing topology**

Only 4 real Ayase stations (5 for Naka) exist as potential graph nodes. A GCN/GAT
over a single-digit-node graph, with no encoded upstream/downstream edge structure
(river_station orders HEC-RAS cross-sections within a reach, not the monitoring
stations across the network), cannot produce a scientifically meaningful message-
passing result — any number reported would be an artifact of a toy graph, not
evidence about river network structure. Unblocking needs both more monitored nodes
and an actual hydrological edge list (which station drains into which), neither of
which currently exists in this repo.

---

## 5. Proposed architecture (extends, does not replace, `src/aquanexus/`)

Following the existing `config → hecras/data → ml → api → dashboard` dependency rule
(`docs/ARCHITECTURE.md` "Module boundaries"), new work adds sibling packages under
`ml/`, each depending only on `data` and `ml.models`/`ml.evaluator`, never on `api`:

```
src/aquanexus/
  ml/
    forecasting.py   # Phase 1: lag/rolling/lead helpers, walk-forward splitter — DONE (infra only, result blocked)
    uncertainty.py   # Phase 2: bootstrap + quantile interval models — DONE; Phase 3 conformal wrapper extends this file
    deep/            # Phase 5: MLP, LSTM (PyTorch, new optional extra)
    transfer/        # Phase 6: fine-tune / frozen-extractor strategies over existing holdout
    bayesian/        # Phase 7: Bayesian linear + hierarchical regression
    graph/           # Phase 8: left as interface stubs only, per §4 blocker
  remote_sensing/     # Phase 4: feature-extraction interfaces; blocked pending station geocoding
```

`uncertainty/` is built as a reusable interface (point prediction → interval) from
Phase 2 onward, so Phase 3 (conformal) and Phase 7 (Bayesian credible intervals) can
implement the same interface and be compared like-for-like, per the original request.

---

## 6. Dependencies between phases

```
Phase 2 (uncertainty) ──▶ Phase 3 (conformal) ──▶ Phase 7 (Bayesian, compared against both)
Phase 1 (forecasting, blocked) ──▶ LSTM half of Phase 5
Phase 6 (transfer) depends on the existing cross-river holdout, not on Phase 1/5
Phase 4 (remote sensing) is independent of the above but gated on station geocoding
Phase 8 (GNN) depends on Phase 4 or 6 only for node features; gated on §4's network-size blocker
```

Recommended order given the above (unchanged from the requested sequence, since the
blockers do not reorder it — they just mean Phase 1 and the LSTM part of Phase 5 ship
as documented-blocked infrastructure rather than results):

1. Phase 1 — build infrastructure, mark forecasting result blocked, do not fabricate.
2. Phase 2 — uncertainty (viable, real result expected).
3. Phase 3 — conformal (viable, with calibration-set-size caveat stated).
4. Phase 4 — remote sensing (source station coordinates first; blocked until then).
5. Phase 5 — MLP viable-but-likely-loses; LSTM blocked with Phase 1.
6. Phase 6 — transfer learning proper (viable, builds on existing zero-shot result).
7. Phase 7 — Bayesian modeling (viable), compared against Phase 2/3 intervals.
8. Phase 8 — GNN interface stubs only; do not report a message-passing result off a
   single-digit-node graph.

## 7. Scientific risks carried into every later phase

- n=138 (Ayase) / n=192 (Naka) is small for *any* additional model family; every new
  technique must be benchmarked against the existing Ridge/persistence baseline
  (RMSE 1.785 / 1.818, R² 0.394 / 0.385), not against the synthetic-HSI R² 0.99.
- Station count (4/5) limits both spatial generalization claims and any graph-based
  method — this is a hard ceiling set by the monitoring program, not by modeling
  effort.
- Every phase must keep using the grouped/station-held-out protocol already
  established; a random split on this dataset is known-leaky
  (`ml/splits.py::random_split` docstring) and must not be reintroduced by a new
  model's default CV.

No code was changed in this audit (§1–§7 above are Phase 0 as originally written).
Phase 1 has since been implemented as infrastructure with a blocked result — see the
status update in §4 and `docs/EXPERIMENTS.md` EXP-001. Phase 2 (uncertainty) has been
implemented with real, viable results — see the status update in §4 and
`docs/EXPERIMENTS.md` EXP-002. Phase 3 onward has not started.
