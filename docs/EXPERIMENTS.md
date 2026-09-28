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
