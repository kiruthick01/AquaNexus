"""Bayesian modeling (Phase 7): pooled and hierarchical linear regression.

## Three intervals, three different claims

Easy to conflate, so stated precisely here since this module produces two of
the three:

* **Credible interval** (Bayesian, parameter-level): a posterior probability
  statement about a *parameter* - "given the data and the prior, there is a
  90% posterior probability the true coefficient lies in this range."
  Exposed here via :meth:`BayesianLinearModel.posterior_summary`. It is not
  about where a new observation will fall.
* **Prediction interval** (frequentist, `ml.uncertainty`) / **posterior
  predictive interval** (Bayesian, this module's `predict_interval`): about
  *one new observation*. This module's version marginalises over posterior
  parameter uncertainty *and* adds the model's own observation-noise term
  (``sigma``) per posterior draw, so it is a genuine prediction interval in
  the same sense `BootstrapIntervalModel` and `QuantileIntervalModel` are -
  directly comparable to them, unlike a credible interval on a coefficient.
* **Conformal interval** (`SplitConformalModel`): no probability model at
  all - a distribution-free coverage guarantee under exchangeability. Compare
  its empirical coverage against this module's posterior predictive coverage
  as two different routes to the same *kind* of claim (an interval for a new
  observation), built on entirely different assumptions (a likelihood model
  here; only exchangeability there).

## Priors

Both models use weakly-informative priors chosen from the physical scale of
the problem (the observed spread of the target and of standardised
features), not tuned against any held-out score - doing the latter would be
exactly the leakage this project's validation rules forbid elsewhere.

## Models

:class:`BayesianLinearModel` - pooled, no group structure, the direct
Bayesian analogue of the canonical Ridge model.

:class:`BayesianHierarchicalModel` - partial pooling of the intercept by
station: each station gets its own intercept, but those intercepts are drawn
from a shared population distribution whose spread is itself estimated from
the data, rather than either forcing all stations to share one intercept
(complete pooling) or fitting each station in total isolation (no pooling).
For a station not seen during fitting - the canonical leave-one-station-out
check - prediction correctly draws a fresh intercept from the *population*
distribution, not from any observed station's fitted value; reusing an
observed station's intercept for an unseen station is a common mistake this
implementation avoids deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger
from aquanexus.ml.uncertainty import IntervalPrediction

log = get_logger("ml.bayesian")


def _require_pymc():
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PyMC is required for aquanexus.ml.bayesian. Install the 'bayesian' "
            "optional extra: pip install 'aquanexus[bayesian]'"
        ) from exc
    return pm


@dataclass
class PosteriorSummary:
    """Per-parameter posterior mean/sd and credible interval at ``level``."""

    table: pd.DataFrame
    level: float


def _standardize_fit(X: pd.DataFrame, feature_names: list[str]):
    raw = X[feature_names].to_numpy(dtype=float)
    median = np.nanmedian(raw, axis=0)
    raw = np.where(np.isnan(raw), median, raw)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std[std == 0] = 1.0
    return (raw - mean) / std, median, mean, std


def _standardize_apply(X: pd.DataFrame, feature_names: list[str], median, mean, std):
    raw = X[feature_names].to_numpy(dtype=float)
    raw = np.where(np.isnan(raw), median, raw)
    return (raw - mean) / std


class BayesianLinearModel:
    """Pooled Bayesian linear regression.

    Priors:

    - ``alpha ~ Normal(y.mean(), 2 * y.std())``: intercept centred at the
      observed target mean, wide enough (2 standard deviations) to be weakly
      informative rather than fixing the intercept near a guessed value.
    - ``beta_j ~ Normal(0, 5)`` for each standardised feature: a standard
      weakly-informative default for regression coefficients once inputs are
      on a unit scale (Gelman et al.'s general recommendation) - wide enough
      to allow an effect many times larger than physically plausible,
      without being improper/flat.
    - ``sigma ~ HalfNormal(2 * y.std())``: residual scale, weakly informative
      and scaled to the target's own observed variability, not to an
      arbitrary constant.
    """

    def __init__(self, draws: int = 1000, tune: int = 1000, chains: int = 4,
                seed: int = 42, output_range: tuple[float, float] | None = None):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.seed = seed
        self.output_range = output_range
        self.trace = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, groups=None) -> BayesianLinearModel:
        pm = _require_pymc()

        self._feature_names = list(X.columns)
        X_std, self._median, self._mean, self._std = _standardize_fit(X, self._feature_names)
        y_arr = np.asarray(y, dtype=float)
        n_features = X_std.shape[1]

        with pm.Model() as model:
            alpha = pm.Normal("alpha", mu=y_arr.mean(), sigma=2 * y_arr.std())
            beta = pm.Normal("beta", mu=0.0, sigma=5.0, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=2 * y_arr.std())
            mu = alpha + pm.math.dot(X_std, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_arr)
            self.trace = pm.sample(self.draws, tune=self.tune, chains=self.chains,
                                   random_seed=self.seed, progressbar=False)
        self.model = model
        return self

    def posterior_summary(self, level: float = 0.9) -> PosteriorSummary:
        import arviz as az

        summary = az.summary(self.trace, ci_prob=level, ci_kind="hdi",
                             var_names=["alpha", "beta", "sigma"])
        return PosteriorSummary(table=summary, level=level)

    def predict_interval(self, X: pd.DataFrame, level: float = 0.9) -> IntervalPrediction:
        if self.trace is None:
            raise RuntimeError("model is not fitted")

        X_std = _standardize_apply(X, self._feature_names, self._median, self._mean, self._std)
        alpha_samples = self.trace.posterior["alpha"].to_numpy().reshape(-1)
        beta_samples = self.trace.posterior["beta"].to_numpy().reshape(-1, len(self._feature_names))
        sigma_samples = self.trace.posterior["sigma"].to_numpy().reshape(-1)

        mu = alpha_samples[:, None] + beta_samples @ X_std.T  # (n_draws, n_rows)
        rng = np.random.default_rng(self.seed)
        noise = rng.normal(0.0, 1.0, size=mu.shape) * sigma_samples[:, None]
        posterior_predictive = mu + noise

        point = posterior_predictive.mean(axis=0)
        alpha_level = 1.0 - level
        lower = np.quantile(posterior_predictive, alpha_level / 2.0, axis=0)
        upper = np.quantile(posterior_predictive, 1.0 - alpha_level / 2.0, axis=0)

        if self.output_range is not None:
            low, high = self.output_range
            point = np.clip(point, low, high)
            lower = np.clip(lower, low, high)
            upper = np.clip(upper, low, high)
        return IntervalPrediction(point=point, lower=lower, upper=upper, level=level)


class BayesianHierarchicalModel:
    """Partial pooling of the intercept by station; slopes shared across stations.

    Priors:

    - ``mu_alpha ~ Normal(y.mean(), 2 * y.std())``: hyperprior mean intercept
      across stations - the population the station intercepts are drawn from.
    - ``sigma_alpha ~ HalfNormal(y.std())``: hyperprior spread of station
      intercepts. This is *estimated from the data*, not fixed by hand - it
      is what lets the model decide, from the data itself, how much stations
      genuinely differ, rather than assuming an amount of heterogeneity in
      advance.
    - ``alpha_station ~ Normal(mu_alpha, sigma_alpha)``, one per observed
      station: partially pooled toward the population mean, shrunk more for
      a station with fewer observations.
    - ``beta_j ~ Normal(0, 5)`` for each standardised feature: shared across
      stations, the same weakly-informative default as the pooled model.
    - ``sigma ~ HalfNormal(2 * y.std())``: shared residual scale.
    """

    def __init__(self, draws: int = 1000, tune: int = 1000, chains: int = 4,
                seed: int = 42, output_range: tuple[float, float] | None = None):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.seed = seed
        self.output_range = output_range
        self.trace = None
        self._feature_names: list[str] = []
        self._station_names: np.ndarray = np.array([])

    def fit(self, X: pd.DataFrame, y: pd.Series, groups) -> BayesianHierarchicalModel:
        pm = _require_pymc()

        self._feature_names = list(X.columns)
        X_std, self._median, self._mean, self._std = _standardize_fit(X, self._feature_names)
        y_arr = np.asarray(y, dtype=float)
        station_codes, station_names = pd.factorize(pd.Series(groups).reset_index(drop=True))
        self._station_names = station_names.to_numpy()
        n_features = X_std.shape[1]
        n_stations = len(station_names)

        with pm.Model() as model:
            mu_alpha = pm.Normal("mu_alpha", mu=y_arr.mean(), sigma=2 * y_arr.std())
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=y_arr.std())
            # Non-centered parameterization: with only n_stations=4 groups, the
            # centered form (alpha_station ~ Normal(mu_alpha, sigma_alpha)
            # directly) couples each station's intercept tightly to
            # sigma_alpha, producing exactly the "funnel" geometry NUTS
            # struggles with at small group counts (observed as sampler
            # divergences during development). Sampling a standard-normal
            # offset and scaling it separately removes that coupling without
            # changing the prior this represents - same distribution, a
            # numerically better-behaved parameterization of it.
            z_station = pm.Normal("z_station", mu=0.0, sigma=1.0, shape=n_stations)
            alpha_station = pm.Deterministic("alpha_station", mu_alpha + z_station * sigma_alpha)
            beta = pm.Normal("beta", mu=0.0, sigma=5.0, shape=n_features)
            sigma = pm.HalfNormal("sigma", sigma=2 * y_arr.std())
            mu = alpha_station[station_codes] + pm.math.dot(X_std, beta)
            pm.Normal("obs", mu=mu, sigma=sigma, observed=y_arr)
            self.trace = pm.sample(self.draws, tune=self.tune, chains=self.chains,
                                   random_seed=self.seed, progressbar=False)
        self.model = model
        return self

    def predict_interval(self, X: pd.DataFrame, level: float = 0.9,
                         station: str | None = None) -> IntervalPrediction:
        """Posterior predictive interval.

        If ``station`` names a station this model was fitted on, that
        station's own partially-pooled intercept is used. Otherwise (the
        default, and the only correct choice for a genuinely unseen
        station - `docs/BAYESIAN_MODELING.md`'s leave-one-station-out check),
        a fresh intercept is drawn from the *population* distribution
        (``mu_alpha``, ``sigma_alpha``) once per posterior sample - not
        copied from any observed station.
        """
        if self.trace is None:
            raise RuntimeError("model is not fitted")

        X_std = _standardize_apply(X, self._feature_names, self._median, self._mean, self._std)
        beta_samples = self.trace.posterior["beta"].to_numpy().reshape(-1, len(self._feature_names))
        sigma_samples = self.trace.posterior["sigma"].to_numpy().reshape(-1)
        n_draws = beta_samples.shape[0]
        rng = np.random.default_rng(self.seed)

        if station is not None and station in self._station_names:
            station_idx = int(np.where(self._station_names == station)[0][0])
            alpha_samples = (self.trace.posterior["alpha_station"]
                            .to_numpy()[:, :, station_idx].reshape(-1))
        else:
            mu_alpha_samples = self.trace.posterior["mu_alpha"].to_numpy().reshape(-1)
            sigma_alpha_samples = self.trace.posterior["sigma_alpha"].to_numpy().reshape(-1)
            alpha_samples = rng.normal(mu_alpha_samples, sigma_alpha_samples)

        mu = alpha_samples[:, None] + beta_samples @ X_std.T
        noise = rng.normal(0.0, 1.0, size=(n_draws, X_std.shape[0])) * sigma_samples[:, None]
        posterior_predictive = mu + noise

        point = posterior_predictive.mean(axis=0)
        alpha_level = 1.0 - level
        lower = np.quantile(posterior_predictive, alpha_level / 2.0, axis=0)
        upper = np.quantile(posterior_predictive, 1.0 - alpha_level / 2.0, axis=0)

        if self.output_range is not None:
            low, high = self.output_range
            point = np.clip(point, low, high)
            lower = np.clip(lower, low, high)
            upper = np.clip(upper, low, high)
        return IntervalPrediction(point=point, lower=lower, upper=upper, level=level)
