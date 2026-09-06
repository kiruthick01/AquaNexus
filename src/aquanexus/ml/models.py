"""Model definitions and the prediction wrapper.

Follows ML_STRATEGY.md §6: XGBoost as the primary model, Random Forest and
linear regression as baselines, plus a persistence-style constant predictor that
establishes the floor any real model must clear.

Hyperparameters are those in §6.1. Two adjustments, both forced by the dataset
being far smaller than the 50,000 rows the spec assumed:

* ``early_stopping_rounds`` moved into the constructor. XGBoost 2.x removed it
  from ``fit()``; passing it there raises ``TypeError``.
* Depth is capped more tightly than §6.1's 7 when the training set is small,
  because a 7-deep tree on a few thousand grouped rows memorises groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("ml.models")

ModelType = Literal["xgboost", "random_forest", "linear", "mean"]


@dataclass
class ModelConfig:
    """Configuration for one model."""

    model_type: ModelType = "xgboost"
    random_seed: int = settings.RANDOM_SEED
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int | None = 20
    extra: dict[str, Any] = field(default_factory=dict)


class MeanPredictor:
    """Predicts the training mean for everything.

    The floor. A model that cannot beat this has learned nothing, and reporting
    R² without it invites reading 0.4 as a success when it is barely above
    predicting a constant.
    """

    def __init__(self) -> None:
        self.value = 0.0

    def fit(self, X, y, **_):  # noqa: N803 - sklearn signature
        self.value = float(np.mean(y))
        return self

    def predict(self, X):  # noqa: N803
        return np.full(len(X), self.value)


class ModelFactory:
    """Creates estimators from a :class:`ModelConfig`."""

    @staticmethod
    def create(config: ModelConfig):
        if config.model_type == "xgboost":
            import xgboost as xgb

            params = dict(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                learning_rate=config.learning_rate,
                subsample=config.subsample,
                colsample_bytree=config.colsample_bytree,
                min_child_weight=1,
                gamma=0,
                reg_alpha=config.reg_alpha,
                reg_lambda=config.reg_lambda,
                random_state=config.random_seed,
                objective="reg:squarederror",
                eval_metric="rmse",
                n_jobs=-1,
            )
            # XGBoost 2.x takes this in the constructor, not in fit().
            if config.early_stopping_rounds:
                params["early_stopping_rounds"] = config.early_stopping_rounds
            params.update(config.extra)
            return xgb.XGBRegressor(**params)

        if config.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=max(config.max_depth * 2, 10),
                random_state=config.random_seed,
                n_jobs=-1,
                **config.extra,
            )

        if config.model_type == "linear":
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            # Ridge rather than plain least squares: the feature set contains
            # near-collinear pairs by construction (flow area with top width, DO
            # with its deficit), which makes an unregularised fit unstable.
            #
            # Imputation and scaling are part of the estimator, not a separate
            # preprocessing step, so they are fitted on the training partition
            # only. Fitting them on the whole frame would leak test statistics
            # into training. The tree models need neither - they split on NaN
            # natively and are scale-invariant.
            return make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                Ridge(alpha=1.0, random_state=config.random_seed, **config.extra),
            )

        if config.model_type == "mean":
            return MeanPredictor()

        raise ValueError(f"unknown model type {config.model_type!r}")


class HabitatPredictor:
    """Wraps an estimator with feature bookkeeping and bounded predictions."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.model = ModelFactory.create(self.config)
        self.feature_names: list[str] = []
        self.is_fitted = False

    # -- training -----------------------------------------------------------

    def fit(self, X: pd.DataFrame, y, validation: tuple | None = None):  # noqa: N803
        self.feature_names = list(X.columns)

        if self.config.model_type == "xgboost":
            if validation is not None and len(validation[1]):
                X_val, y_val = validation
                self.model.fit(X, y, eval_set=[(X_val[self.feature_names], y_val)],
                               verbose=False)
            else:
                # Early stopping is configured on the constructor in XGBoost 2.x,
                # and it then *requires* an eval set at fit time - fitting without
                # one raises "Must have at least 1 validation dataset for early
                # stopping". Turn it off rather than fabricating a holdout.
                self.model.set_params(early_stopping_rounds=None)
                self.model.fit(X, y)
        else:
            self.model.fit(X, y)

        self.is_fitted = True
        log.info("fitted %s on %d rows x %d feature(s)",
                 self.config.model_type, len(X), len(self.feature_names))
        return self

    # -- inference ----------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Predict HSI, clipped to [0, 1].

        Clipping is not cosmetic: HSI is a bounded index, and a regressor will
        happily return 1.03 near the top of its range. An out-of-range
        suitability score is meaningless to anyone reading it.
        """
        self._require_fitted()
        return np.clip(self.model.predict(X[self.feature_names]), 0.0, 1.0)

    def predict_with_uncertainty(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
        """Predictions and a per-row spread.

        Only tree ensembles carry a usable spread - the disagreement between
        member trees. For anything else the spread is NaN rather than a
        fabricated constant, so callers cannot mistake silence for confidence.
        """
        self._require_fitted()
        predictions = self.predict(X)
        rows = X[self.feature_names]

        estimators = getattr(self.model, "estimators_", None)
        if estimators is not None:  # random forest
            per_tree = np.stack([e.predict(rows) for e in estimators])
            return predictions, per_tree.std(axis=0)

        if self.config.model_type == "xgboost":
            # Predict with a truncated ensemble at several sizes; the spread
            # across those is a proxy for how settled the prediction is.
            total = self.model.get_booster().num_boosted_rounds()
            cuts = [max(1, int(total * f)) for f in (0.25, 0.5, 0.75, 1.0)]
            staged = np.stack([
                self.model.predict(rows, iteration_range=(0, c)) for c in cuts
            ])
            return predictions, staged.std(axis=0)

        return predictions, np.full(len(X), np.nan)

    # -- feature importance --------------------------------------------------

    def importances(self) -> pd.Series:
        """Native feature importance, descending. Empty if unsupported."""
        self._require_fitted()
        model = self.model
        # Unwrap a pipeline to reach the estimator that carries the weights.
        if hasattr(model, "steps"):
            model = model.steps[-1][1]
        values = getattr(model, "feature_importances_", None)
        if values is None:
            coefficients = getattr(model, "coef_", None)
            if coefficients is None:
                return pd.Series(dtype=float)
            values = np.abs(np.ravel(coefficients))
        return pd.Series(values, index=self.feature_names).sort_values(ascending=False)

    # -- persistence ---------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        import joblib

        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "config": self.config,
                     "feature_names": self.feature_names}, path)
        log.info("saved model to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> HabitatPredictor:
        import joblib

        payload = joblib.load(Path(path))
        predictor = cls.__new__(cls)
        predictor.config = payload["config"]
        predictor.model = payload["model"]
        predictor.feature_names = payload["feature_names"]
        predictor.is_fitted = True
        return predictor

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
