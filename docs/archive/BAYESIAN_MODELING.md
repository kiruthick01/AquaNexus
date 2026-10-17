# Bayesian Modeling (Phase 7)

## Three intervals, three different claims

Restated precisely because this phase produces two of the three and compares
against the third:

- **Credible interval** (Bayesian, parameter-level): a posterior probability
  statement about a *parameter* - "given the data and the prior, there is a
  90% posterior probability the true coefficient lies in this range." Not
  about where a new observation falls. Available via
  `BayesianLinearModel.posterior_summary()`.
- **Prediction interval** (`ml.uncertainty`'s bootstrap/quantile methods) /
  **posterior predictive interval** (this module): about *one new
  observation*, incorporating both parameter uncertainty and the model's own
  observation-noise term. This is what `predict_interval` returns here, and
  it is directly comparable to the frequentist prediction intervals in
  `docs/UNCERTAINTY.md` - same kind of claim, different machinery underneath.
- **Conformal interval** (`SplitConformalModel`): no probability model at
  all - a distribution-free coverage guarantee under exchangeability.

## Priors, and why

Both models (`src/aquanexus/ml/bayesian.py`) use weakly-informative priors
sized from the target's own observed scale, chosen before looking at any
held-out result and never adjusted to improve one:

- Intercept(s): `Normal(y.mean(), 2 * y.std())` - centred at the observed
  mean, wide enough to be uninformative relative to the data's own spread.
- Coefficients (standardised features): `Normal(0, 5)` - Gelman et al.'s
  standard weakly-informative default once inputs are on a unit scale, wide
  enough to permit an implausibly large effect without being flat.
- Residual scale: `HalfNormal(2 * y.std())` - weakly informative, scaled to
  the target's own variability.
- Hierarchical model only: `sigma_alpha ~ HalfNormal(y.std())`, the
  population spread of station intercepts, *estimated from the data* rather
  than fixed - this is what makes the pooling "partial" rather than a
  hand-chosen amount of shrinkage.

## A numerical note: non-centered parameterization

Initial development runs of the hierarchical model produced sampler
divergences (4-13 per fold, on 3 of 4 leave-one-station-out folds) - the
well-documented "funnel" pathology that a *centered* hierarchical
parameterization (`alpha_station ~ Normal(mu_alpha, sigma_alpha)` sampled
directly) produces when there are only a few groups (here, 4 stations).
Reparameterizing to sample a standard-normal offset and scale it separately
(`alpha_station = mu_alpha + z_station * sigma_alpha`, `z_station ~
Normal(0,1)`) removed the divergences in the final run below. This changes
nothing about the prior or the model being fit - it is the same distribution,
sampled in a numerically better-behaved way - and is a standard fix, not a
metric-motivated one.

## Validation protocol

Leave-one-station-out, identical to `ModelValidator` and Phase 2/3's
protocol, so results are directly comparable without recomputing bootstrap
or conformal. For the hierarchical model, a held-out station is - correctly
- never in the training set, so its prediction draws a fresh intercept from
the population hyperprior (`mu_alpha`, `sigma_alpha`) rather than reusing any
observed station's fitted value (`BayesianHierarchicalModel.predict_interval`
docstring; verified by `tests/test_bayesian.py`).

## Results (real data, `scripts/phase7_bayesian_experiment.py`, target 90% posterior predictive interval)

| Method | Coverage | Mean width | RMSE | R² |
|---|---|---|---|---|
| Bayesian linear (pooled) | 0.913 | 7.475 | 2.612 | -0.297 |
| Bayesian hierarchical | 0.971 | 9.299 | 2.741 | -0.428 |
| Bootstrap (`docs/UNCERTAINTY.md`) | 0.920 | 6.738 | 1.794 | 0.389 |
| Quantile regression (`docs/UNCERTAINTY.md`) | 0.703 | 3.411 | 1.632 | 0.494 |

**Per-station (Bayesian):**

| Model | Station | n | Coverage | Mean width |
|---|---|---|---|---|
| Pooled | 52内匠橋 | 48 | 0.958 | 12.022 |
| Pooled | 54槐戸橋 | 36 | 0.944 | 4.255 |
| Pooled | 55畷橋 | 36 | 0.833 | 5.938 |
| Pooled | 57綾瀬川合流点前 | 18 | 0.889 | 4.865 |
| Hierarchical | 52内匠橋 | 48 | 0.979 | 13.686 |
| Hierarchical | 54槐戸橋 | 36 | 0.972 | 6.480 |
| Hierarchical | 55畷橋 | 36 | 0.972 | 7.983 |
| Hierarchical | 57綾瀬川合流点前 | 18 | 0.944 | 5.873 |

## Interpretation

**Both Bayesian models achieve reasonable-to-conservative coverage** (0.913
pooled, 0.971 hierarchical, against a 0.90 target) **but substantially worse
point-prediction accuracy than Ridge** (RMSE 2.612/2.741 vs Ridge's 1.785,
R² actually negative for both). This gap was investigated rather than
reported blind: fit on the full dataset, the Bayesian pooled model's
in-sample RMSE (1.328) matches Ridge's (1.330) almost exactly, and every
parameter's `r_hat` is 1.00 (no convergence problem). The posterior mean
coefficients, though, are substantially larger in magnitude than Ridge's
tuned ones (e.g. one coefficient posterior mean of 2.17 against Ridge's
1.584 for the same feature; another at -1.8 against -0.371) - the `Normal(0,
5)` prior is a much weaker regularizer than Ridge's cross-validated
`alpha=1.0` penalty at n=138 with 10 partially collinear features
(`ML_METHODOLOGY.md`'s documented collinear pairs). The Bayesian model fits
the training data just as well as Ridge, but that weaker regularization lets
it fit more of each fold's idiosyncratic noise, which costs it more on the
held-out station than Ridge's tighter shrinkage does. **This is not a bug
and the priors were not adjusted to fix it** - adjusting `Normal(0,5)` to
something tighter after seeing this gap would be exactly the "choosing
priors because they improve metrics" the project's methodology forbids. The
finding itself - that a generic weakly-informative prior under-regularizes
relative to a specifically cross-validated Ridge penalty, at this sample
size and this much collinearity - is the answer to what this phase asked.

**The hierarchical model is more conservative (wider, higher coverage) and
scores slightly worse on point accuracy than the pooled model.** With 4
stations, partial pooling has very little data to distinguish genuine
station-to-station variation from noise the pooled model's fixed intercept
would just have to average over; the hyperprior's own uncertainty
(`sigma_alpha`) adds another layer of variance the pooled model does not
carry, widening intervals for exactly the reason partial pooling is supposed
to be conservative when there is little group-level evidence.

**Against the non-Bayesian methods**: bootstrap achieves better coverage-
for-width than either Bayesian model (narrower at 6.738 while still hitting
0.920 coverage) with far better point accuracy (RMSE 1.794, R² 0.389).
Quantile regression is sharper still but badly undercovers (already
documented in `docs/UNCERTAINTY.md`). On this dataset, the bootstrap
interval around Ridge remains the strongest interval method measured across
all of Phase 2/3/7.

## Limitations

- A single, generic weakly-informative prior family was used; a prior
  informed by domain knowledge of plausible dissolved-oxygen sensitivity to
  each specific feature (rather than a generic `Normal(0,5)` on every
  standardised coefficient) might close some of the regularization gap with
  Ridge - not attempted here, since constructing such a prior without
  reference to this dataset's own fitted coefficients is a substantial
  domain-elicitation exercise beyond this phase's scope.
- 4 stations is a small number of groups for a hierarchical model to
  partially pool over; `sigma_alpha`'s own posterior is itself
  poorly constrained by so few groups, which is part of why the hierarchical
  model's intervals widen rather than sharpen relative to the pooled one.
- MCMC ran without a compiled backend (`g++` unavailable in this
  environment; PyTensor's numba fallback was used), which slowed sampling
  but did not fail it - `r_hat` values were checked and were at the target of
  1.00 throughout the diagnostic run.

## Conclusion

Bayesian modeling here provides calibrated, if wide, uncertainty (question 9
in `docs/ML_ROADMAP.md`'s closing list) but does not provide a competitive
*point* prediction at this sample size, because its generic weakly-informative
prior regularizes less aggressively than Ridge's tuned penalty. The
frequentist bootstrap interval around Ridge remains the project's best
uncertainty method measured to date; Bayesian modeling's contribution here is
a genuine, understood answer for why, not a new best model.
