"""Transfer learning / domain adaptation (Phase 6): Ayase (source) to Naka (target).

The existing cross-river holdout (`scripts/holdout_river.py`,
`docs/HOLDOUT_RIVER.md`) already measures **zero-shot transfer**: the Ayase
Ridge model applied to the Naka, completely unchanged, no target data used at
all (pooled R² -0.081). This module implements what that result does not:
strategies that actually use a little target-domain data, or target-domain
feature statistics, to adapt - and names each one precisely, since the spec
this project follows is explicit that ordinary retraining must not be called
"transfer learning".

Four distinct claims, each a different amount of target-domain information:

* **Zero-shot** (existing, cited here, not reproduced): no target data at
  all, of any kind.
* **Domain-aligned** (:func:`domain_aligned_predict`): uses target *feature*
  statistics (mean/std of the target's own inputs) but no target *labels*.
  The source model's learned coefficients are kept entirely frozen; only the
  standardisation the inputs pass through before reaching them changes.
* **Fine-tuned** (:func:`fine_tune_ridge`): uses a small amount of target
  *labelled* data, but regularises the adapted coefficients toward the
  source model's own coefficients (an L2-SP-style biased-regression
  estimator - Xuhong et al. 2018) rather than fitting a fresh model from
  scratch. As the shrinkage strength grows, the fit degenerates exactly to
  the unchanged source model; as it shrinks to zero, it degenerates to an
  ordinary independent fit on the target slice alone - which is precisely
  why this is not that, at any shrinkage strength above zero.
* **Frozen-feature-extractor + target head** (:class:`FrozenHeadAdapter`):
  uses the same small labelled target slice, but adapts a neural network by
  retraining only its final linear layer, holding every earlier layer's
  weights fixed. Built on `ml.deep.MLPModel`, whose own in-domain point
  accuracy already loses to Ridge (`docs/DEEP_LEARNING.md`) - this method is
  included because the spec asks for it and the technique is worth
  evaluating even on a weaker base model, not because the base model is
  expected to win.

None of the source/target test infrastructure here refits or reruns the
existing zero-shot holdout; that result stays exactly as
`docs/HOLDOUT_RIVER.md` reports it, reproducible independently of this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger
from aquanexus.ml.models import HabitatPredictor

log = get_logger("ml.transfer")


def domain_aligned_predict(source_model: HabitatPredictor, X_target: pd.DataFrame
                          ) -> np.ndarray:
    """Predict on target data using the source model's coefficients, but the
    *target's own* feature mean/std instead of the source's.

    Uses no target label at all - only the target inputs' own distribution.
    The idea: if a feature is shifted in level between rivers (e.g. the Naka
    reads warmer/cleaner throughout - `HOLDOUT_RIVER.md`'s documented bias),
    re-centring it on the target's own mean before it reaches the frozen
    source coefficients may remove some of that shift's effect on the
    prediction, without touching what the model learned about the
    *relationship* between features and dissolved oxygen.
    """
    pipeline = source_model.model
    imputer = pipeline.named_steps["simpleimputer"]
    ridge = pipeline.named_steps["ridge"]
    feature_names = source_model.feature_names

    imputed = imputer.transform(X_target[feature_names])

    target_mean = np.nanmean(imputed, axis=0)
    target_std = np.nanstd(imputed, axis=0)
    target_std[target_std == 0] = 1.0
    aligned = (imputed - target_mean) / target_std

    predictions = ridge.intercept_ + aligned @ ridge.coef_
    if source_model.config.output_range is not None:
        low, high = source_model.config.output_range
        predictions = np.clip(predictions, low, high)
    return predictions


def fine_tune_ridge(source_model: HabitatPredictor, X_adapt: pd.DataFrame, y_adapt,
                    shrinkage: float = 10.0) -> tuple[np.ndarray, float]:
    """Refit Ridge coefficients on a small target-domain slice, regularised
    toward the source model's own coefficients rather than toward zero.

    Solves ``argmin_b ||y - Xb||^2 + shrinkage * ||b - b_source||^2``
    analytically: ``b = (X^T X + shrinkage * I)^-1 (X^T y + shrinkage * b_source)``.
    The intercept is left unregularised (fit as the mean residual after
    applying the shrunk coefficients), standard practice for ridge-family
    estimators, so the model is free to correct a level shift between rivers
    without that correction being penalised the same way a slope change is.

    ``shrinkage`` controls how much the target slice is trusted: large values
    keep the fit close to the unchanged source model (appropriate when the
    target slice is very small, as it is here); ``shrinkage=0`` recovers an
    ordinary independent least-squares fit on the target slice alone, which
    is a different, unregularised estimator - not what this function returns
    unless explicitly asked for that degenerate case.

    Returns ``(coefficients, intercept)`` in the source model's standardised
    feature space - callers must transform inputs with the *source* model's
    own imputer/scaler (not the target's) before applying them, since the
    coefficients being adapted are defined in that space.
    """
    pipeline = source_model.model
    imputer = pipeline.named_steps["simpleimputer"]
    scaler = pipeline.named_steps["standardscaler"]
    ridge = pipeline.named_steps["ridge"]
    feature_names = source_model.feature_names

    X = scaler.transform(imputer.transform(X_adapt[feature_names]))
    y = np.asarray(y_adapt, dtype=float)

    b_source = ridge.coef_
    n_features = X.shape[1]
    XtX = X.T @ X
    Xty = X.T @ y
    coefficients = np.linalg.solve(XtX + shrinkage * np.eye(n_features),
                                   Xty + shrinkage * b_source)
    intercept = float(np.mean(y - X @ coefficients))
    return coefficients, intercept


def predict_fine_tuned(source_model: HabitatPredictor, X: pd.DataFrame,
                       coefficients: np.ndarray, intercept: float) -> np.ndarray:
    """Predict with :func:`fine_tune_ridge`'s output, in the source model's
    standardised feature space."""
    pipeline = source_model.model
    imputer = pipeline.named_steps["simpleimputer"]
    scaler = pipeline.named_steps["standardscaler"]
    feature_names = source_model.feature_names

    X_std = scaler.transform(imputer.transform(X[feature_names]))
    predictions = intercept + X_std @ coefficients
    if source_model.config.output_range is not None:
        low, high = source_model.config.output_range
        predictions = np.clip(predictions, low, high)
    return predictions


class FrozenHeadAdapter:
    """Freeze an `ml.deep.MLPModel`'s learned layers; retrain only a new
    final linear head on a small target-domain slice.

    Built on the Phase 5 MLP, whose own in-domain accuracy already loses to
    Ridge (`docs/DEEP_LEARNING.md`) - included because the spec explicitly
    asks for this strategy to be evaluated, not because the base network is
    expected to win. A failure here is informative (frozen-extractor transfer
    does not rescue a weak base model), not a bug.
    """

    def __init__(self, source_mlp, learning_rate: float = 1e-2, max_epochs: int = 200,
                patience: int = 20, seed: int = 42):
        self.source_mlp = source_mlp
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self._head = None

    def fit(self, X_adapt: pd.DataFrame, y_adapt) -> FrozenHeadAdapter:
        from aquanexus.ml.deep import TrainingCurve, _require_torch, set_seed

        torch = _require_torch()
        from torch import nn

        set_seed(self.seed)
        base_model = self.source_mlp.torch_module
        for param in base_model.parameters():
            param.requires_grad_(False)

        # The frozen body's output is base_model's last Linear layer's input
        # width - reach it by running a forward pass through every layer but
        # the final one.
        body_layers = list(base_model.children())[:-1]
        body = nn.Sequential(*body_layers)
        head_in = base_model[-1].in_features
        self._head = nn.Linear(head_in, 1)

        X_norm = self.source_mlp.normalize(X_adapt)
        X_t = torch.from_numpy(X_norm)
        y_t = torch.from_numpy(np.asarray(y_adapt, dtype=np.float32).reshape(-1, 1))

        optimizer = torch.optim.Adam(self._head.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.curve = TrainingCurve()
        best_state, best_loss, stale = None, float("inf"), 0
        body.eval()
        epoch = 0
        for epoch in range(self.max_epochs):  # noqa: B007 - epochs_run reads it after the loop
            with torch.no_grad():
                features = body(X_t)
            self._head.train()
            optimizer.zero_grad()
            loss = loss_fn(self._head(features), y_t)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.item())
            self.curve.train_loss.append(loss_value)

            if loss_value < best_loss - 1e-6:
                best_loss = loss_value
                best_state = {k: v.clone() for k, v in self._head.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self._head.load_state_dict(best_state)
        self.curve.epochs_run = epoch + 1
        self._body = body
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        from aquanexus.ml.deep import _require_torch

        torch = _require_torch()
        X_norm = self.source_mlp.normalize(X)
        with torch.no_grad():
            features = self._body(torch.from_numpy(X_norm))
            predictions = self._head(features).numpy().reshape(-1)
        return predictions

    def save(self, path):
        from aquanexus.ml.serialization import save_model
        return save_model(self, path)

    @classmethod
    def load(cls, path) -> FrozenHeadAdapter:
        from aquanexus.ml.serialization import load_model
        return load_model(cls, path)
