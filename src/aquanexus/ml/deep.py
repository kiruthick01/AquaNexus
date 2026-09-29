"""Small neural-network models (Phase 5): an MLP baseline and an LSTM.

PyTorch, per ``docs/ML_ROADMAP.md`` Phase 5 ("use PyTorch unless the
repository already has a justified deep-learning framework" - it does not).
It is declared as a new optional extra (``pyproject.toml`` ``deep``), not a
core dependency, since nothing outside this module needs it.

Both models here are deliberately small. At n~100-140 real training rows
(the canonical dissolved-oxygen dataset), a network with more parameters than
training points is in exactly the overfitting regime
``ML_METHODOLOGY.md`` already documented for XGBoost - "start small" is not
a stylistic preference, it is what this sample size can support at all.

:class:`MLPModel` is a point predictor, evaluated under the same
station-held-out protocol as every other model in this project
(``scripts/phase5_deep_learning_experiment.py``).

:class:`LSTMModel` is **infrastructure only, not run against real data**.
Phase 1 (``docs/EXPERIMENTS.md`` EXP-001) found zero usable lag-complete
forecasting rows at the current sampling density, so there is no real
sequence for it to train on. It is tested against synthetic sequences only
(``tests/test_deep.py``), so that if Phase 1's blocker clears later, the
mechanism does not need to be built from scratch.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("ml.deep")


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "PyTorch is required for aquanexus.ml.deep. Install the 'deep' "
            "optional extra: pip install 'aquanexus[deep]'"
        ) from exc
    return torch


try:
    import torch.nn as _nn
except ImportError:  # pragma: no cover - exercised only without the extra
    _nn = None

if _nn is not None:
    class _LSTMNet(_nn.Module):
        """A named, module-level class - not one nested inside a method -
        because `pickle` (and so `joblib`, `LSTMModel.save`) can only
        serialize a class it can re-import by qualified name; a class
        defined inside a function has no such name."""

        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = _nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.head = _nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])


def set_seed(seed: int) -> None:
    """Seed every RNG torch touches, so two fits with the same seed produce
    bit-identical weights, not merely similar ones."""
    torch = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class TrainingCurve:
    """Per-epoch loss history plus what early stopping decided."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = 0
    epochs_run: int = 0
    training_seconds: float = 0.0


class MLPModel:
    """Small feed-forward network for point prediction of a scalar target.

    Two hidden layers (16, 8 units) by default, with dropout and weight
    decay - regularised deliberately, not left to early stopping alone, given
    how little training data backs each fold. Checkpointing keeps the
    best-validation-loss weights, not the last epoch's, since the last epoch
    at this sample size is often already overfitting.
    """

    def __init__(self, hidden_sizes: tuple[int, ...] = (16, 8), dropout: float = 0.2,
                 learning_rate: float = 1e-2, weight_decay: float = 1e-3,
                 max_epochs: int = 300, patience: int = 20, seed: int = 42,
                 output_range: tuple[float, float] | None = None):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.output_range = output_range
        self.curve = TrainingCurve()
        self._model = None
        self._feature_names: list[str] = []
        self._median = None
        self._mean = None
        self._std = None

    def _build(self, n_features: int):
        _require_torch()
        from torch import nn

        sizes = [n_features, *self.hidden_sizes]
        layers: list = []
        for a, b in zip(sizes[:-1], sizes[1:], strict=True):
            layers += [nn.Linear(a, b), nn.ReLU(), nn.Dropout(self.dropout)]
        layers.append(nn.Linear(sizes[-1], 1))
        return nn.Sequential(*layers)

    def _normalize(self, X: pd.DataFrame) -> np.ndarray:
        values = X[self._feature_names].to_numpy(dtype=np.float32)
        values = np.where(np.isnan(values), self._median, values)
        return (values - self._mean) / self._std

    def fit(self, X: pd.DataFrame, y, X_val: pd.DataFrame | None = None,
           y_val=None) -> MLPModel:
        torch = _require_torch()
        set_seed(self.seed)

        self._feature_names = list(X.columns)
        raw = X[self._feature_names].to_numpy(dtype=np.float32)
        self._median = np.nanmedian(raw, axis=0)
        raw = np.where(np.isnan(raw), self._median, raw)
        self._mean = raw.mean(axis=0)
        self._std = raw.std(axis=0)
        self._std[self._std == 0] = 1.0
        X_norm = (raw - self._mean) / self._std
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            X_val_norm = self._normalize(X_val)
            y_val_arr = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
        else:
            log.warning("no validation set supplied; early stopping will monitor "
                       "training loss, which cannot detect overfitting")

        self._model = self._build(X_norm.shape[1])
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        X_t = torch.from_numpy(X_norm)
        y_t = torch.from_numpy(y_arr)
        if has_val:
            X_val_t = torch.from_numpy(X_val_norm)
            y_val_t = torch.from_numpy(y_val_arr)

        self.curve = TrainingCurve()
        best_state, best_val, best_epoch, stale = None, float("inf"), 0, 0
        start = time.perf_counter()
        epoch = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            optimizer.zero_grad()
            loss = loss_fn(self._model(X_t), y_t)
            loss.backward()
            optimizer.step()
            self.curve.train_loss.append(float(loss.item()))

            self._model.eval()
            with torch.no_grad():
                val_loss = (float(loss_fn(self._model(X_val_t), y_val_t).item())
                           if has_val else float(loss.item()))
            self.curve.val_loss.append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val, best_epoch = val_loss, epoch
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self.curve.best_epoch = best_epoch
        self.curve.epochs_run = epoch + 1
        self.curve.training_seconds = time.perf_counter() - start
        log.info("MLP: %d epoch(s), best at %d (val loss %.4f), %.2fs, %d parameter(s)",
                 self.curve.epochs_run, self.curve.best_epoch, best_val,
                 self.curve.training_seconds, self.n_parameters())
        return self

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    @property
    def torch_module(self):
        """The fitted `torch.nn.Sequential` body - exposed so a frozen-layer
        transfer method (`ml.transfer.FrozenHeadAdapter`) can reuse the
        learned layers without duplicating them."""
        return self._model

    def normalize(self, X: pd.DataFrame) -> np.ndarray:
        """Apply this model's fitted imputation/standardisation to ``X``,
        without predicting - the same public need as :attr:`torch_module`."""
        return self._normalize(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        torch = _require_torch()
        X_norm = self._normalize(X)
        self._model.eval()
        with torch.no_grad():
            pred = self._model(torch.from_numpy(X_norm)).numpy().reshape(-1)
        if self.output_range is not None:
            low, high = self.output_range
            pred = np.clip(pred, low, high)
        return pred

    def save(self, path):
        from aquanexus.ml.serialization import save_model
        return save_model(self, path)

    @classmethod
    def load(cls, path) -> MLPModel:
        from aquanexus.ml.serialization import load_model
        return load_model(cls, path)


class LSTMModel:
    """Minimal single-layer LSTM sequence regressor - infrastructure only.

    Takes a pre-normalised 3-D array ``(n_sequences, seq_len, n_features)``
    and predicts one scalar per sequence from the final timestep's hidden
    state. Deliberately does not duplicate `MLPModel`'s imputation/scaling
    (a caller with a real sequence dataset would need its own, sequence-aware
    version of that, which does not exist because no real sequence dataset
    exists yet - see the module docstring).
    """

    def __init__(self, input_size: int, hidden_size: int = 8, num_layers: int = 1,
                 learning_rate: float = 1e-2, weight_decay: float = 1e-3,
                 max_epochs: int = 200, patience: int = 15, seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.curve = TrainingCurve()
        self._model = None

    def _build(self):
        _require_torch()
        return _LSTMNet(self.input_size, self.hidden_size, self.num_layers)

    def fit(self, X_seq: np.ndarray, y, X_val_seq: np.ndarray | None = None,
           y_val=None) -> LSTMModel:
        torch = _require_torch()
        set_seed(self.seed)

        X_t = torch.from_numpy(np.asarray(X_seq, dtype=np.float32))
        y_t = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
        has_val = X_val_seq is not None and y_val is not None and len(X_val_seq) > 0
        if has_val:
            X_val_t = torch.from_numpy(np.asarray(X_val_seq, dtype=np.float32))
            y_val_t = torch.from_numpy(np.asarray(y_val, dtype=np.float32).reshape(-1, 1))

        self._model = self._build()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        loss_fn = torch.nn.MSELoss()

        self.curve = TrainingCurve()
        best_state, best_val, best_epoch, stale = None, float("inf"), 0, 0
        start = time.perf_counter()
        epoch = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            optimizer.zero_grad()
            loss = loss_fn(self._model(X_t), y_t)
            loss.backward()
            optimizer.step()
            self.curve.train_loss.append(float(loss.item()))

            self._model.eval()
            with torch.no_grad():
                val_loss = (float(loss_fn(self._model(X_val_t), y_val_t).item())
                           if has_val else float(loss.item()))
            self.curve.val_loss.append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val, best_epoch = val_loss, epoch
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self.curve.best_epoch = best_epoch
        self.curve.epochs_run = epoch + 1
        self.curve.training_seconds = time.perf_counter() - start
        return self

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self._model.parameters())

    def predict(self, X_seq: np.ndarray) -> np.ndarray:
        torch = _require_torch()
        self._model.eval()
        with torch.no_grad():
            pred = self._model(
                torch.from_numpy(np.asarray(X_seq, dtype=np.float32))
            ).numpy().reshape(-1)
        return pred

    def save(self, path):
        from aquanexus.ml.serialization import save_model
        return save_model(self, path)

    @classmethod
    def load(cls, path) -> LSTMModel:
        from aquanexus.ml.serialization import load_model
        return load_model(cls, path)
