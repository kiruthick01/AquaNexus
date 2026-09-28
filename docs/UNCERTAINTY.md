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

## Limitations

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
- Neither method has been evaluated under domain shift (cross-river) yet;
  that check is Phase 3's, which reuses the existing Naka holdout
  infrastructure (`HOLDOUT_RIVER.md`) specifically to test whether interval
  coverage degrades under distribution shift.
