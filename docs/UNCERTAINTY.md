# Predictive Uncertainty (Phase 2)

## Terminology, precisely

Four terms are easy to conflate and mean different things. Every claim below
names which one it is making.

| Term | Question it answers | Behaviour as data grows |
|---|---|---|
| **Confidence interval** | Where is the true *parameter* (e.g. the mean prediction at this x)? | Narrows toward zero width |
| **Prediction interval** | Where will *one new observation* at this x fall? | Stays wide - must always cover residual scatter, not just parameter uncertainty |
| **Model / epistemic uncertainty** | How much would the fitted relationship change on a different sample of this size? | Shrinks with more data |
| **Data / aleatoric uncertainty** | Irreducible scatter in y given x (measurement noise, unmeasured drivers) | Does not shrink with more data |

Everything this phase reports is a **prediction interval**: "if this reading
were taken again, where would it likely fall", not "where is the true mean
DO for this condition". The two methods below build that interval two
different ways.

A narrower, pre-existing method in this codebase, `HabitatPredictor.
predict_with_uncertainty` (`ml/models.py`), answers a different and smaller
question - how much do ensemble members (trees) disagree - has no coverage
guarantee, and returns `NaN` for Ridge, the model that actually wins on this
dataset. It is not superseded by this work; the two answer different
questions and should not be conflated.

## Methods

Implemented in `src/aquanexus/ml/uncertainty.py`, both behind the same
interface (`IntervalModel.fit` / `.predict_interval` → `IntervalPrediction`)
so Phase 3 (conformal) and Phase 7 (Bayesian credible intervals) can be
compared like-for-like later.

**A. Bootstrap (`BootstrapIntervalModel`).** 500 case-resampled refits of
Ridge give the spread of point predictions across plausible refits - model
uncertainty. A residual, drawn from out-of-fold (station-held-out) residuals
within the training partition, is added to each bootstrap prediction before
taking quantiles - injecting data/noise uncertainty on top, rather than
reporting the model-uncertainty spread alone and calling it a prediction
interval.

**B. Quantile regression (`QuantileIntervalModel`).** Linear quantile
regressors (pinball loss, `sklearn.linear_model.QuantileRegressor`) fit
directly at the 5th, 50th, and 95th percentiles. No bootstrap, no residual
pool - the interval is whatever the quantile fits say.

## Validation protocol

Canonical: leave-one-station-out (`scripts/phase2_uncertainty_experiment.py`),
identical grouping to `ModelValidator`. For the bootstrap method, the residual
pool for each fold is itself computed by a *nested* leave-one-station-out
inside that fold's three training stations, so the held-out station never
touches training, residual estimation, or fitting for its own fold.

## Results (real data, `data/processed/ayase_do_dataset.csv`, n=138, target 90% PI)

| Method | Pooled coverage | Mean width (mg/L) | RMSE | R² | MAE |
|---|---|---|---|---|---|
| Bootstrap | **0.920** | 6.738 | 1.794 | 0.389 | 1.283 |
| Quantile regression | **0.703** | 3.411 | 1.632 | 0.494 | 1.152 |

Bootstrap RMSE (1.794) reproduces the canonical Ridge baseline (1.785,
`ML_METHODOLOGY.md`) closely, as expected - the bootstrap point prediction is
the mean of 500 Ridge refits on the same protocol.

**Per-station:**

| Method | Station | n | Coverage | Mean width | MAE |
|---|---|---|---|---|---|
| Bootstrap | 52内匠橋 | 48 | 0.938 | 8.038 | 1.015 |
| Bootstrap | 54槐戸橋 | 36 | 0.972 | 5.365 | 0.918 |
| Bootstrap | 55畷橋 | 36 | 0.861 | 7.012 | 1.989 |
| Bootstrap | 57綾瀬川合流点前 | 18 | 0.889 | 5.466 | 1.313 |
| Quantile | 52内匠橋 | 48 | 0.521 | 1.857 | 0.950 |
| Quantile | 54槐戸橋 | 36 | 0.917 | 4.377 | 0.855 |
| Quantile | 55畷橋 | 36 | 0.611 | 3.844 | 1.719 |
| Quantile | 57綾瀬川合流点前 | 18 | 0.944 | 4.759 | 1.156 |

## Interpretation

**Bootstrap is calibrated but not sharp.** Pooled coverage (0.920) sits close
to the 0.90 target, and no station falls badly short (worst is 0.861 on
n=36). The cost is width: a mean 6.74 mg/L interval against an observed range
of roughly 3-17 mg/L (`ML_METHODOLOGY.md`) covers well over a third of the
entire observed range. At n=138 this is the honest price of a calibrated
interval, not a defect to tune away - reporting a narrower interval here
would mean reporting one that has not been shown to be trustworthy.

**Quantile regression is sharp but not calibrated.** Half the width (3.41 vs
6.74) but pooled coverage of 0.703 against a 0.90 target - a real, measured
failure, most severe at 52内匠橋 (0.521, barely better than a coin flip for a
90% claim) and 55畷橋 (0.611). Linear quantile regression at n≈90-105 training
rows (three stations) does not have enough data to pin down the 5th and 95th
percentiles reliably; the fitted quantile lines are themselves high-variance
estimates. No quantile crossing was observed in this run
(`QuantileIntervalModel` logs a warning if it occurs), so the miscalibration
is a genuine estimation-variance problem, not a crossing artifact.

**Point-prediction accuracy is not the finding here.** Quantile regression's
median prediction posts a lower RMSE (1.632) and higher R² (0.494) than the
bootstrap/Ridge baseline (1.794 / 0.389) - a byproduct of optimizing pinball
loss rather than squared error on a small, noisy dataset, not evidence that
quantile regression is the better *point* predictor. The canonical
point-prediction comparison remains `ML_METHODOLOGY.md`'s Ridge result; this
phase's contribution is the interval, not a new point-prediction champion.

## Recommendation

At the current sample size, the bootstrap method is the one whose stated
coverage can actually be trusted; the quantile-regression interval should not
be used for a calibrated claim until either more data is available or its
undercoverage is specifically corrected for (e.g. by conformalizing it -
Phase 3 exists partly to do this in a distribution-free way rather than by
tuning the quantile regressor's regularisation against the same small
evaluation set, which would leak).

## Limitations (Phase 2)

- n=138 across 4 stations means every coverage number above is itself
  estimated from as few as 18 test points (57綾瀬川合流点前) - a single
  additional miss or hit shifts that station's coverage by ~5.6 percentage
  points. Station-level coverage numbers should be read as indicative, not
  precise.
- The bootstrap residual pool, when a station's own nested CV cannot be run
  (fewer than 3 remaining training groups), falls back to in-sample residuals
  and logs a warning - this does not occur in the current 4-station Ayase
  protocol (each fold leaves exactly 3 training stations) but would if a
  future dataset had 3 or fewer stations total.
- Neither method had been evaluated under domain shift (cross-river) at this
  point; that check is Phase 3's, below.

---

## Phase 3: split conformal prediction

**Method** (`SplitConformalModel`, `ml/uncertainty.py`): hold out a random
25% calibration slice of the training rows, fit Ridge on the rest, score the
calibration slice with absolute residuals, and take the interval as point ±
the finite-sample-corrected quantile of those residuals
(`ceil((n+1)(1-alpha))/n`, not the naive empirical quantile - the correction
is what gives the *marginal* coverage guarantee under exchangeability). The
resulting interval has constant width across every row, since the score is
unconditional - see the class docstring for why, and for the exact
assumptions this method depends on.

**Why this phase exists:** the guarantee is conditional on calibration and
test data being *exchangeable*. This project's canonical station-held-out
and cross-river protocols each break that assumption in a different, known
way, so running the same conformal procedure through all three is a direct
measurement of what that violation costs - not a hypothetical concern.

### Results (real data, `scripts/phase3_conformal_experiment.py`, target 90% coverage)

| Regime | n test | Coverage | Mean width (mg/L) | MAE |
|---|---|---|---|---|
| A. In-domain (random split, same distribution) | 28 | 0.857 | 4.720 | 0.888 |
| B. Held-out station (leave-one-station-out, pooled) | 138 | 0.870 | 6.145 | 1.314 |
| C. Cross-river (Naka, pooled) | 192 | 0.849 | 5.957 | 1.665 |

**Held-out station, per station:**

| Station | n | Coverage | Mean width |
|---|---|---|---|
| 52内匠橋 | 48 | 0.979 | 7.188 |
| 54槐戸橋 | 36 | 0.972 | 5.696 |
| 55畷橋 | 36 | 0.583 | 4.775 |
| 57綾瀬川合流点前 | 18 | 0.944 | 7.003 |

**Cross-river (Naka), per station:**

| Station | n | Coverage | Mean width |
|---|---|---|---|
| 46八条橋 | 48 | 0.479 | 5.957 |
| 48豊橋 | 36 | 0.972 | 5.957 |
| 49松富橋 | 36 | 0.972 | 5.957 |
| 50行幸橋 | 36 | 0.944 | 5.957 |
| 51道橋 | 36 | 1.000 | 5.957 |

### Interpretation

**Pooled coverage looks deceptively stable across regimes** - 0.857, 0.870,
0.849 - which at first reads as "conformal coverage barely degrades under
distribution shift", the opposite of what `HOLDOUT_RIVER.md` found for point
predictions (pooled zero-shot R² -0.081). It is not that finding overturned;
it is the same failure mode this project has documented before, now visible
in the per-station table instead of the pooled one: **46八条橋 - the one Naka
station on a different sub-reach carrying up to 145 m³/s, twice the Ayase's
training maximum (`HOLDOUT_RIVER.md`) - covers at 0.479, barely better than
chance for a 90% claim**, while the four in-range Naka stations all cover at
0.94-1.00. The pooled cross-river number (0.849) averages a station the
interval has no business claiming 90% coverage on with four where it happens
to. Exactly the same shape appears within the Ayase itself: 55畷橋 covers at
0.583 while the other three Ayase stations cover at 0.94-0.98.

**None of the three pooled numbers actually reaches the 0.90 target**,
including in-domain (0.857). At n_test=28 for the in-domain regime, this gap
is plausibly single-split sampling noise rather than a systematic shortfall
- 24/28 successes is not a surprising draw if the true rate is 0.90 - and a
single conformal run at small n does not distinguish the two. This is stated
as an open question, not resolved: repeating regime A over multiple random
calibration/test splits would be needed to tell noise from a real
shortfall, and was not done here to keep this run's assumptions and n_test
simple and auditable.

**The in-domain per-station table is not reported above** because its
per-station counts are as low as n=1 (one station's entire representation in
a 28-row random test slice), making a per-station coverage number there
meaningless rather than merely noisy; it is visible in the script's log
output for transparency but should not be read as a finding.

### Assumptions, restated from the code

- Coverage is **marginal**, not conditional: it is a guarantee about the
  average over draws of calibration and test data, not about any specific
  test point or subgroup. The per-station breakdowns above are exactly why
  this distinction matters in practice - the marginal number can look fine
  while a subgroup is badly miscovered.
- The interval is unconditional in width (constant across every row in a
  given fit), because the nonconformity score used here is plain absolute
  residual. A locally-adaptive score would let width vary with local
  difficulty; not implemented here.
- Exchangeability is the assumption every regime beyond A tests the cost of
  violating. It does not hold, by construction, whenever calibration and
  test data come from different stations or rivers.

### Limitations (Phase 3)

- Single random seed, single split per regime - coverage numbers, especially
  for the n_test=28 in-domain regime, carry meaningful sampling variance of
  their own that this run does not quantify.
- `calibration_fraction=0.25` was fixed in advance rather than tuned; tuning
  it against these same evaluation numbers would leak.
- The out-of-range Naka station's poor coverage was expected from
  `HOLDOUT_RIVER.md` before this experiment ran (extrapolation is already
  known to break point predictions there); this phase's contribution is
  showing the interval fails in the same place, not discovering a new
  failure mode.
