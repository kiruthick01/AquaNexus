# Experiment Log

One entry per experiment, in implementation order. Machine-readable results (real
numbers only, never fabricated) are tracked in `experiments/results.csv`; this
file gives each row its scientific narrative.

---

## EXP-001 — Phase 1: DO(t+1)/(t+3)/(t+7) forecasting from the real record

**Question:** Can dissolved oxygen at a monitoring station be forecast 1, 3, or
7 days ahead from its own recent history (lag features) using the real,
non-synthetic observation record?

**Hypothesis:** If the record were dense and regular enough, lag/rolling
features plus a walk-forward split should let a simple model beat a persistence
baseline at short horizons, degrading as the horizon grows. This experiment
tests the *precondition* for that hypothesis, not the hypothesis itself.

**Dataset:** `data/processed/ayase_do_dataset.csv` (138 rows, 4 stations,
2022-04-07 to 2025-03-07), built by
`aquanexus.data.dataset.build_water_quality_dataset()`.

**Features:** `dissolved_oxygen_lag{1,3,7}d`, built by
`aquanexus.ml.forecasting.add_lag_features` with a relative tolerance of ±20%
of the nominal lag (floored at 1 day) — see that module's docstring for why a
relative rather than fixed tolerance is required at these horizons.

**Model:** None fit. See Results.

**Validation:** `aquanexus.ml.forecasting.walk_forward_splits`, chronological,
grouped by station, no shuffling; leakage checked with
`assert_no_temporal_leakage`.

**Results:** Running `scripts/phase1_forecasting_feasibility.py` against the
real dataset:

| Horizon | Usable rows (lag-complete AND target-in-tolerance) |
|---|---|
| t+1 | 0 / 138 |
| t+3 | 0 / 138 |
| t+7 | 0 / 138 |

Walk-forward split construction itself succeeds (20 fold-objects obtainable
across the 4 stations at 5 folds each — the splitter is not the blocker), but
every fold's usable rows are removed once the lag+lead completeness filter is
applied. No MAE/RMSE/R² is reported because no model was trained.

**Interpretation:** The record is irregular grab sampling (roughly monthly,
gaps of weeks to months), not a regular series. A "1-day lag" cannot be
honestly built when the nearest real prior sample is typically weeks away.
This is a data-density problem, not a feature-engineering or modelling
problem — see `docs/DATA_LIMITATIONS.md` for the derived minimum
(~50-100 regularly-spaced observations per station over 2+ seasons) that
would unblock this.

**Limitations:** The tolerance (±20% of nominal lag, floored at 1 day) is a
documented modelling choice, not derived from any statistical test; a looser
tolerance would manufacture more "usable" rows at the cost of the lag no
longer meaning what it claims to mean, which would defeat the purpose of this
check.

**Conclusion:** Phase 1 forecasting is **blocked by data availability** on the
current record. The temporal infrastructure (lag/lead/rolling features,
walk-forward splitting, leakage detection) is implemented and tested against
synthetic series (`tests/test_forecasting.py`) and is ready to use without
further engineering if a denser, regularly-sampled record becomes available.
No forecasting result is reported, and none should be inferred from this
entry.

---

## EXP-002 — Phase 2: prediction intervals for dissolved oxygen

**Question:** Can a prediction interval around the dissolved-oxygen point
forecast be produced, and does its stated coverage hold under the canonical
station-held-out protocol?

**Hypothesis:** A bootstrap interval (model uncertainty + injected residual
noise) should achieve closer-to-nominal coverage than a directly-fit quantile
regression at this sample size, at the cost of width, because quantile
regression's 5th/95th percentile fits are themselves high-variance estimates
with only ~100 training rows.

**Dataset:** `data/processed/ayase_do_dataset.csv` (138 rows, 4 stations),
same as EXP-001.

**Features:** `aquanexus.data.dataset.DO_FEATURES` (unchanged from the
canonical Ridge baseline).

**Model:** `aquanexus.ml.uncertainty.BootstrapIntervalModel` (Ridge, 500
resamples) and `QuantileIntervalModel` (linear pinball-loss regression at
q=0.05/0.5/0.95), targeting a 90% prediction interval.

**Validation:** Leave-one-station-out, matching `ModelValidator`; the
bootstrap method's residual pool is itself computed by a nested
leave-one-station-out inside each fold's three training stations, so the
held-out station never enters residual estimation for its own fold.

**Results:** `scripts/phase2_uncertainty_experiment.py`, pooled over all 4
stations:

| Method | Coverage (target 0.90) | Mean width | RMSE | R² |
|---|---|---|---|---|
| Bootstrap | 0.920 | 6.738 mg/L | 1.794 | 0.389 |
| Quantile regression | 0.703 | 3.411 mg/L | 1.632 | 0.494 |

Full per-station breakdown in `docs/UNCERTAINTY.md`.

**Interpretation:** The hypothesis held. Bootstrap achieves near-nominal
coverage but a wide interval (6.74 mg/L against an observed range of roughly
3-17 mg/L). Quantile regression is sharper (half the width) but
substantially undercovers (0.703 against a 0.90 target, as low as 0.521 at
one station) - an overconfident interval that should not be trusted at this
sample size without further correction.

**Limitations:** Station-level coverage is estimated from as few as 18 test
points, so per-station numbers carry wide uncertainty of their own. Neither
method has been checked under cross-river domain shift yet (Phase 3).

**Conclusion:** A calibrated prediction interval is achievable on this
dataset (bootstrap), but it is wide; a sharper interval (quantile regression)
is not currently trustworthy at this sample size. Report the bootstrap
interval when a calibrated claim is required, and do not present the
quantile-regression interval as calibrated without further work.

---

## EXP-003 — Phase 3: does conformal prediction's coverage guarantee survive distribution shift?

**Question:** Split conformal prediction guarantees marginal coverage under
exchangeability. Does that guarantee visibly degrade when calibration and
test data are drawn from different stations (spatial shift) or different
rivers (the project's existing, real domain-shift benchmark)?

**Hypothesis:** Pooled coverage should degrade moving from in-domain, to
held-out-station, to cross-river; and any degradation should be worst at the
specific subgroup already known to be out of the model's training range
(`HOLDOUT_RIVER.md`'s 46八条橋, at up to 145 m³/s against the Ayase's 74).

**Dataset:** Ayase canonical dataset (138 rows, 4 stations) for training and
calibration in all three regimes; Naka holdout dataset (192 rows, 5 stations,
`data/processed/holdout_naka.json`) as the cross-river test set in regime C.

**Features:** `DO_FEATURES`, unchanged.

**Model:** `aquanexus.ml.uncertainty.SplitConformalModel` (Ridge point model,
25% random calibration slice, absolute-residual nonconformity score,
finite-sample-corrected quantile), target 90% coverage.

**Validation:** Three regimes, `scripts/phase3_conformal_experiment.py`:
(A) random 80/20 in-domain split of the Ayase data; (B) leave-one-station-out
across the 4 Ayase stations, pooled and per-station; (C) train+calibrate on
all of Ayase, test on the independently reserved Naka holdout.

**Results:**

| Regime | n test | Coverage | Mean width | MAE |
|---|---|---|---|---|
| A. In-domain | 28 | 0.857 | 4.720 | 0.888 |
| B. Held-out station (pooled) | 138 | 0.870 | 6.145 | 1.314 |
| C. Cross-river Naka (pooled) | 192 | 0.849 | 5.957 | 1.665 |

Full per-station tables in `docs/UNCERTAINTY.md`.

**Interpretation:** The hypothesis was half right. Pooled coverage does *not*
degrade much across regimes (0.857 → 0.870 → 0.849) - on its own, a
misleadingly reassuring result. The per-station breakdown shows why it is
misleading: 46八条橋 (the one out-of-training-range Naka station) covers at
0.479, and 55畷橋 (the Ayase's own hardest station in every prior phase)
covers at 0.583, while every other station in both regimes covers at
0.94-1.00. The pooled number averages a station the interval fails on with
several it happens to work on - the same shape `HOLDOUT_RIVER.md` already
found for point predictions, now shown to affect interval coverage too.

**Limitations:** Single seed, single split per regime; the in-domain regime's
n_test=28 makes its 0.857 vs the 0.90 target hard to distinguish from
sampling noise without repeated splits, which were not run. The
out-of-range failure was predicted in advance from `HOLDOUT_RIVER.md`, not
discovered by this experiment.

**Conclusion:** Conformal prediction's marginal coverage guarantee held
reasonably well *on average* across all three regimes, but that average hid
a real, specific failure at exactly the subgroup already known to be outside
the model's training range. A pooled coverage number is not sufficient to
certify a conformal interval as trustworthy under distribution shift; the
per-subgroup breakdown is required, and was not optional in this case.

---

## EXP-004 — Phase 4: can remote-sensing features be extracted for any real station?

**Question:** Can NDVI/NDWI/MNDWI or other Sentinel-2/Landsat-derived
features be computed for the Ayase or Naka monitoring stations?

**Hypothesis:** None can be computed, because no station coordinate is on
record anywhere in this repository - this is a data-prerequisite check, not
a hypothesis about remote sensing itself.

**Dataset:** Real station name lists from the Ayase (4 stations) and Naka
(5 stations) canonical datasets.

**Features:** N/A - no imagery was fetched.

**Model:** N/A.

**Validation:** `aquanexus.remote_sensing.require_station_coordinates`, run
against the real combined 9-station list via
`scripts/phase4_remote_sensing_feasibility.py`.

**Results:** 9/9 stations raise `MissingStationCoordinatesError`. Zero
stations have a usable coordinate.

**Interpretation:** The hypothesis held completely. The feature-extraction
machinery itself (spectral indices, cloud filtering, spatial buffers,
temporal matching) is implemented and tested against synthetic rasters
(`tests/test_remote_sensing.py`), but cannot be exercised on any real
AquaNexus station until a citable coordinate source is integrated - see
`docs/REMOTE_SENSING.md` and `docs/DATA_LIMITATIONS.md`.

**Limitations:** This experiment tests only the coordinate prerequisite, not
satellite data availability, cloud cover, or feature predictive value - none
of those can be assessed without coordinates first.

**Conclusion:** Phase 4 is **blocked by data availability** (station
coordinates), confirmed against the real station list rather than assumed.
No remote-sensing feature, and no ablation result, is reported for this
phase, and none should be inferred from this entry.

---

## EXP-005 — Phase 5: does an MLP outperform classical ML for dissolved oxygen?

**Question:** Does a small feed-forward neural network beat the canonical
Ridge model, or any other classical baseline, at predicting dissolved oxygen?

**Hypothesis:** No - consistent with `ML_METHODOLOGY.md`'s existing finding
that XGBoost already overfits at n=138, an MLP (more parameters, no
inductive bias toward linearity) should do no better, and plausibly worse.

**Dataset:** `data/processed/ayase_do_dataset.csv` (138 rows, 4 stations),
same as every other point-prediction result in this project.

**Features:** `DO_FEATURES`, unchanged.

**Model:** `aquanexus.ml.deep.MLPModel` - 2 hidden layers (16, 8 units),
dropout 0.2, Adam, early stopping (patience 20, max 300 epochs), 321
parameters.

**Validation:** Leave-one-station-out (`ModelValidator`'s protocol), with an
additional random 20% validation slice carved from each fold's three
training stations purely for early stopping, never touching the held-out
station.

**Results:** `scripts/phase5_deep_learning_experiment.py`:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge | 1.785 | 1.274 | 0.394 |
| Persistence | 1.818 | 1.213 | 0.385 |
| Random Forest | 1.884 | 1.425 | 0.325 |
| XGBoost | 1.909 | 1.408 | 0.307 |
| Mean (floor) | 2.294 | 1.779 | 0.000 |
| **MLP** | **2.358** | **1.904** | **-0.057** |

**Interpretation:** The hypothesis held, more strongly than expected: the
MLP does not merely lose to Ridge, it loses to predicting the training mean
(R² below zero). At this sample size, a 321-parameter network has more
capacity than the ~90-120 training rows per fold can constrain, and
regularisation (dropout, weight decay, early stopping) narrows but does not
close that gap.

**Limitations:** A single architecture was evaluated, not a hyperparameter
search (searching against this same n=138 set would leak). The LSTM half of
this phase was not run against real data - see EXP-001; it is implemented
and tested against synthetic sequences only.

**Conclusion:** Deep learning does not outperform classical ML for this
dataset. The canonical dissolved-oxygen model remains Ridge, unchanged by
this phase. Added model complexity did not produce a scientifically
meaningful improvement here - the opposite happened, and is reported as such.

---

## EXP-006 — Phase 6: does using target-domain information improve on zero-shot cross-river transfer?

**Question:** The existing zero-shot result (Ayase Ridge, unchanged, applied
to the Naka: pooled R² -0.081, `docs/HOLDOUT_RIVER.md`) uses no target
information at all. Does using a little - target feature statistics, or a
small labelled target slice - improve on it, and does the amount/kind of
target information used change whether it helps or hurts?

**Hypothesis:** Domain alignment (feature statistics only) should give a
small, safe improvement by correcting the documented Naka/Ayase level shift.
Fine-tuning and frozen-head adaptation, using actual target labels, should do
better still, provided they are adequately regularised against the very
small adaptation slice available (29 rows).

**Dataset:** Ayase (138 rows, source, training only) and Naka (192 rows,
target), split once by row (seed fixed): 29 rows for adaptation, 163 rows
genuinely held out and never used for fitting, calibration, or shrinkage
selection.

**Features:** `DO_FEATURES`, unchanged.

**Model:** Source Ridge and source MLP (Phase 5's, unchanged), adapted by
`aquanexus.ml.transfer`'s four methods.

**Validation:** All four methods scored on the identical 163-row held-out
Naka subset; zero-shot recomputed on this subset (not the full 192 rows) for
a fair comparison.

**Results:** `scripts/phase6_transfer_learning_experiment.py` -
see `docs/TRANSFER_LEARNING.md` for the full table.

Zero-shot R² -0.133, domain-aligned R² -0.089 (small real improvement, no
target labels used). Fine-tuning swept over shrinkage {1, 10, 100, 1000,
10000}: catastrophic at 1/10/100 (worst R² -6.548 at shrinkage=10), then the
**best result of any method tested** at shrinkage=1000 (R² 0.372), slightly
declining again at shrinkage=10000 (R² 0.263). Frozen-head (MLP) scored R²
0.349, essentially matching the best fine-tuned result.

**Interpretation:** The hypothesis was partly right and partly wrong.
Domain alignment helped, as predicted. Fine-tuning's outcome depended
entirely on shrinkage strength: too weak (1-100) relative to the 29-row
adaptation slice's scale, it collapsed to a near-independent overfit on 29
noisy points and performed far worse than doing nothing; strong enough
(1000), it became the single best method in the experiment. Frozen-head
adaptation, whose base MLP loses badly to Ridge in-domain
(`docs/DEEP_LEARNING.md`), matched the best fine-tuned result almost exactly
- both succeed via strong implicit or explicit regularisation of the
adaptation step.

**Limitations:** Single seed, single adaptation/test split, single MLP
architecture. The full shrinkage sweep is reported, not a search that
surfaces only the winning value in isolation.

**Conclusion:** Whether transfer helps or hurts is governed by how strongly
the adaptation is regularised relative to how little target data backs it,
not by which named technique is used. Under-regularised fine-tuning was the
worst outcome measured, worse than zero-shot; adequately-regularised
fine-tuning was the best. See `docs/TRANSFER_LEARNING.md` for the full
sweep and analysis.

---

## EXP-007 — Phase 7: does Bayesian modeling provide useful calibrated uncertainty?

**Question:** Do a Bayesian linear regression and a hierarchical
(partial-pooling-by-station) variant provide calibrated posterior predictive
intervals, and how do they compare to the bootstrap/quantile intervals
already measured (Phase 2) on point accuracy?

**Hypothesis:** Both should achieve close-to-nominal coverage, since their
priors are weakly informative rather than tight. Point accuracy was not
assumed to match Ridge in advance.

**Dataset:** `data/processed/ayase_do_dataset.csv`, same as every other
point-prediction result.

**Features:** `DO_FEATURES`, unchanged.

**Model:** `aquanexus.ml.bayesian.BayesianLinearModel` (pooled) and
`BayesianHierarchicalModel` (station intercepts partially pooled, shared
slopes), NUTS via PyMC, 1000 draws/1000 tune/4 chains per fold.

**Validation:** Leave-one-station-out (`ModelValidator`'s protocol); the
hierarchical model draws a fresh intercept from the population hyperprior for
each held-out station, never reusing an observed station's fitted value.

**Results:** `scripts/phase7_bayesian_experiment.py`:

| Method | Coverage | Mean width | RMSE | R² |
|---|---|---|---|---|
| Bayesian linear (pooled) | 0.913 | 7.475 | 2.612 | -0.297 |
| Bayesian hierarchical | 0.971 | 9.299 | 2.741 | -0.428 |
| Bootstrap (Phase 2) | 0.920 | 6.738 | 1.794 | 0.389 |
| Quantile regression (Phase 2) | 0.703 | 3.411 | 1.632 | 0.494 |

**Interpretation:** The coverage half of the hypothesis held (both Bayesian
methods near or above the 0.90 target). Point accuracy did not match Ridge:
investigated directly (not left unexplained), the Bayesian pooled model's
in-sample fit matches Ridge's almost exactly (RMSE 1.328 vs 1.330,
`r_hat`=1.00 throughout), but its posterior mean coefficients are
substantially larger than Ridge's tuned ones - the generic `Normal(0,5)`
prior regularizes less aggressively than Ridge's cross-validated `alpha=1.0`
penalty at n=138 with collinear features, costing it more on held-out data.
The hierarchical model was more conservative still (wider intervals, slightly
worse RMSE), consistent with 4 stations being a small number of groups to
partially pool over.

**Limitations:** A single generic prior family was used, not one informed by
feature-specific domain knowledge. `docs/BAYESIAN_MODELING.md` documents a
sampling fix (non-centered parameterization) applied to the hierarchical
model to remove divergences encountered during development - a numerical
correction, not a prior change.

**Conclusion:** Bayesian modeling gives calibrated but wide intervals and
does not out-predict Ridge here, for an understood and reported reason
(prior regularization strength, not a modeling error). The bootstrap
interval around Ridge remains the project's best uncertainty method to date.

---

## EXP-008 — Phase 8: can a GNN be trained on the real monitoring-station network?

**Question:** Can a GCN or GAT be trained on the Ayase or Naka monitoring
station network, and would explicit graph structure improve on a non-graph
model?

**Hypothesis:** No GNN can be trained on either real network - Phase 0's
audit found only 4-5 candidate nodes and no upstream/downstream edge list.
This experiment tests that prerequisite directly rather than assuming it.

**Dataset:** Real station name lists from the Ayase (4 stations) and Naka (5
stations) canonical datasets, plus the HEC-RAS sweep's `river_station` field
for both, checked as a candidate (and rejected) topology source.

**Features:** N/A - no graph was trained.

**Model:** N/A.

**Validation:** `aquanexus.ml.graph.require_station_topology`, run against
both real station lists via `scripts/phase8_gnn_feasibility.py`; the
`river_station` field's distinct-value count (49 Ayase, 35 Naka) checked and
confirmed unjoined to any monitoring station identity.

**Results:** 0/2 rivers have a recorded hydrological edge list. `river_station`
orders cross-sections within a reach, not monitoring stations across a
network - confirmed directly, not assumed.

**Interpretation:** The hypothesis held completely. The GNN machinery itself
(dense GCN and GAT layers, node-regression wrappers with correct held-out-node
semantics) is implemented and tested against a synthetic 6-node graph
(`tests/test_graph.py`), but cannot be exercised on either real AquaNexus
network until both blockers - node count and topology - are resolved. See
`docs/GNN.md` and `docs/DATA_LIMITATIONS.md`.

**Limitations:** This experiment tests only the topology and node-count
prerequisites, not whether a real GNN would outperform a non-graph model -
that comparison is meaningless without a real graph to run it on.

**Conclusion:** Phase 8 is **blocked by network size and missing topology**,
confirmed against the real station lists and the one topology-adjacent field
this repository does carry, rather than assumed. No GNN result, and no
graph-vs-non-graph comparison, is reported for this phase, and none should
be inferred from this entry.
