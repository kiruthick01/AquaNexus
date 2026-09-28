"""Predictive uncertainty: point prediction -> prediction interval.

Four terms get used loosely in the literature this module has to be precise
about, because the difference changes what a number is allowed to claim:

* **Confidence interval** - uncertainty about a *parameter* (e.g. "the true
  regression coefficient", or "the mean prediction at this x"). It does not
  say where one new observation will fall; it narrows as data grows, even
  toward zero width.
* **Prediction interval** - uncertainty about *one new observation* at a given
  x. It must stay wide even with infinite data, because it also has to cover
  the residual scatter around the mean, not just uncertainty about the mean
  itself. This is what ``docs/ML_ROADMAP.md`` Phase 2 asks for, and what every
  method in this module returns.
* **Model / epistemic uncertainty** - how much the fitted relationship itself
  would change if fit on a different sample of the same size and origin.
  Shrinks with more data.
* **Data / aleatoric uncertainty** - the irreducible scatter in ``y`` given
  ``x`` (measurement noise, unmeasured drivers). Does not shrink with more
  data.

:class:`BootstrapIntervalModel` builds a genuine prediction interval by
combining both of the latter two explicitly, rather than reporting one and
calling it the other:

1. Case-resampling bootstrap over training rows gives the spread of point
   predictions across plausible refits - **model uncertainty**. Reported
   alone, its quantiles would be a *confidence* interval, not a *prediction*
   interval.
2. Residuals added to each bootstrap prediction inject **data uncertainty**
   - the scatter the model cannot explain even when fit correctly. Wherever a
   grouping column is supplied, these residuals come from held-out
   (station-out) folds, not from the training fit itself, because in-sample
   residuals are optimistic about how much a model actually misses by.

:class:`QuantileIntervalModel` estimates the interval directly by regressing
the conditional quantiles of ``y`` - a different route to the same claim
(prediction interval for one new observation), useful as a check on the
bootstrap method rather than a replacement for it.

Both implement :class:`IntervalModel`, the interface ``docs/ML_ROADMAP.md``
asks be reusable: Phase 3 (conformal) wraps a point model with the same
``fit``/``predict_interval`` shape, and Phase 7 (Bayesian credible intervals)
returns the same :class:`IntervalPrediction`, so all three can be compared on
one coverage/width table - see ``docs/UNCERTAINTY.md``.

Existing, narrower uncertainty in this codebase: ``HabitatPredictor.
predict_with_uncertainty`` (``ml/models.py``) reports the disagreement between
ensemble members for tree models only, with no coverage guarantee and no
result for the model that actually wins on this dataset (Ridge). It answers a
different, narrower question ("how much do the trees disagree") and is not
superseded by this module, which answers "what interval should contain the
next observation, and how often does it actually".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger
from aquanexus.ml.models import HabitatPredictor, ModelConfig

log = get_logger("ml.uncertainty")


@dataclass
class IntervalPrediction:
    """Point prediction plus a two-sided interval at one nominal coverage level."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    level: float  # nominal coverage, e.g. 0.9 for a 90% interval

    def width(self) -> np.ndarray:
        return self.upper - self.lower


class IntervalModel(Protocol):
    """Interface every uncertainty method in this project implements."""

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None) -> IntervalModel: ...  # noqa: E501

    def predict_interval(self, X: pd.DataFrame, level: float = 0.9) -> IntervalPrediction: ...


def _out_of_fold_residuals(X: pd.DataFrame, y: np.ndarray, groups: pd.Series,
                           config: ModelConfig, seed: int) -> np.ndarray:
    """Residuals from held-out-group predictions, not the training fit itself.

    Reusing in-sample residuals to build a prediction interval understates the
    true scatter, because the model was fit to minimise exactly those
    residuals. With too few groups to hold any out (fewer than 3, e.g. a
    single training station left after an outer leave-one-station-out split),
    falls back to in-sample residuals and says so - a documented compromise,
    not a silent one.
    """
    n_groups = groups.nunique()
    if n_groups < 3:
        log.warning(
            "only %d group(s) available for the residual fold; falling back to "
            "in-sample residuals, which understate true prediction noise", n_groups
        )
        model = HabitatPredictor(config).fit(X, y)
        return model.predict(X) - y

    from sklearn.model_selection import GroupKFold

    oof = np.full(len(X), np.nan)
    for train_idx, test_idx in GroupKFold(n_splits=n_groups).split(X, y, groups):
        model = HabitatPredictor(config).fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])
    residuals = oof - y
    return residuals[np.isfinite(residuals)]


class BootstrapIntervalModel:
    """Prediction interval via case-resampling bootstrap + residual injection.

    ``n_bootstrap`` resamples of the training rows are each fit independently;
    at predict time, one residual (drawn with replacement from the
    out-of-fold residual pool) is added to each bootstrap model's prediction,
    and the interval is the empirical quantile range of that combined sample.
    This is a small, explicit procedure rather than a closed-form interval,
    which matters at n~100: normal-theory intervals would assume a residual
    distribution this dataset is too small to actually verify.
    """

    def __init__(self, model_type: str = "linear", n_bootstrap: int = 500,
                 seed: int = 42, output_range: tuple[float, float] | None = None):
        self.model_type = model_type
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.output_range = output_range
        self._bootstrap_models: list[HabitatPredictor] = []
        self._residuals = np.array([])

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
           ) -> BootstrapIntervalModel:
        X = X.reset_index(drop=True)
        y_arr = np.asarray(y, dtype=float)
        config = ModelConfig(model_type=self.model_type, output_range=self.output_range)

        if groups is not None:
            self._residuals = _out_of_fold_residuals(
                X, y_arr, pd.Series(groups).reset_index(drop=True), config, self.seed
            )
        else:
            log.warning("no groups supplied; using in-sample residuals, which "
                       "understate true prediction noise")
            point_model = HabitatPredictor(config).fit(X, y_arr)
            self._residuals = point_model.predict(X) - y_arr

        rng = np.random.default_rng(self.seed)
        n = len(X)
        self._bootstrap_models = []
        for b in range(self.n_bootstrap):
            idx = rng.integers(0, n, size=n)
            resample_config = ModelConfig(model_type=self.model_type,
                                          output_range=self.output_range,
                                          random_seed=self.seed + b)
            model = HabitatPredictor(resample_config).fit(
                X.iloc[idx].reset_index(drop=True), y_arr[idx]
            )
            self._bootstrap_models.append(model)
        return self

    def predict_interval(self, X: pd.DataFrame, level: float = 0.9) -> IntervalPrediction:
        if not self._bootstrap_models:
            raise RuntimeError("model is not fitted")
        if len(self._residuals) == 0:
            raise RuntimeError("no residuals available to build an interval from")

        bootstrap_preds = np.stack([m.predict(X) for m in self._bootstrap_models])
        point = bootstrap_preds.mean(axis=0)

        rng = np.random.default_rng(self.seed + self.n_bootstrap + 1)
        residual_draws = rng.choice(self._residuals, size=bootstrap_preds.shape,
                                    replace=True)
        composite = bootstrap_preds + residual_draws

        alpha = 1.0 - level
        lower = np.quantile(composite, alpha / 2.0, axis=0)
        upper = np.quantile(composite, 1.0 - alpha / 2.0, axis=0)
        return IntervalPrediction(point=point, lower=lower, upper=upper, level=level)


class QuantileIntervalModel:
    """Prediction interval from directly-fit conditional quantile regressions.

    A linear quantile regressor is fit once for the lower tail, once for the
    upper tail (and once at the median for the point prediction), each
    minimising pinball loss at its own quantile. Unlike the bootstrap method,
    this needs no residual pool and no grouping information - the interval is
    whatever the quantile fits say - which makes it a useful independent check
    on the bootstrap interval rather than a like-for-like replacement.
    """

    def __init__(self, alpha: float = 0.01, seed: int = 42, level: float = 0.9):
        #: L1 regularisation strength passed to ``QuantileRegressor``. Held
        #: small and fixed rather than tuned, since tuning it against the
        #: same small dataset used to evaluate coverage would leak.
        self.alpha = alpha
        self.seed = seed
        #: Nominal coverage this instance's quantile pair is fit for. Fixed at
        #: construction, not at predict time - each level needs its own pair
        #: of quantile fits, so switching levels means building a new instance.
        self.level = level
        self._models: dict[float, object] = {}

    def _fit_quantile(self, X: pd.DataFrame, y: np.ndarray, quantile: float):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import QuantileRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            QuantileRegressor(quantile=quantile, alpha=self.alpha, solver="highs"),
        )
        pipeline.fit(X, y)
        return pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None
           ) -> QuantileIntervalModel:
        y_arr = np.asarray(y, dtype=float)
        alpha = 1.0 - self.level
        self._q_low, self._q_high = alpha / 2.0, 1.0 - alpha / 2.0
        self._models = {
            self._q_low: self._fit_quantile(X, y_arr, self._q_low),
            0.5: self._fit_quantile(X, y_arr, 0.5),
            self._q_high: self._fit_quantile(X, y_arr, self._q_high),
        }
        return self

    def predict_interval(self, X: pd.DataFrame, level: float = 0.9) -> IntervalPrediction:
        if not self._models:
            raise RuntimeError("model is not fitted")
        if not np.isclose(level, self.level):
            raise ValueError(
                f"this instance was fit for level={self.level}; level={level} needs "
                "quantiles that were not fit - construct a new instance for a "
                "different level"
            )
        lower = self._models[self._q_low].predict(X)
        upper = self._models[self._q_high].predict(X)
        point = self._models[0.5].predict(X)
        # Pinball-loss fits are not constrained to be monotonic in quantile;
        # an inverted pair at some row is a real finding about this small a
        # dataset, not a bug to silently paper over with np.minimum/maximum.
        inverted = int(np.sum(lower > upper))
        if inverted:
            log.warning("%d/%d row(s) have lower > upper quantile prediction - "
                       "quantile crossing, a known small-sample failure mode",
                       inverted, len(X))
        return IntervalPrediction(point=point, lower=lower, upper=upper, level=level)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def coverage(y_true: np.ndarray, interval: IntervalPrediction) -> float:
    """Fraction of observations that actually fall inside the interval."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= interval.lower) & (y_true <= interval.upper)
    return float(np.mean(inside))


def mean_width(interval: IntervalPrediction) -> float:
    """Average interval width - sharpness. Narrower is better only if coverage
    is also met; a narrow interval that misses is not sharp, it is wrong."""
    return float(np.mean(interval.width()))
