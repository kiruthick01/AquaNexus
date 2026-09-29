"""Tests for the Phase 5 MLP and LSTM models.

MLP tests use small synthetic regression data. LSTM tests use synthetic
sequences only - see ``src/aquanexus/ml/deep.py`` for why it is never run
against real AquaNexus data (Phase 1 forecasting is blocked by data
availability, so no real sequence exists to train it on).

Skipped entirely if the ``deep`` optional extra (PyTorch) is not installed,
consistent with how this project treats every other optional dependency.
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aquanexus.ml.deep import LSTMModel, MLPModel, set_seed  # noqa: E402


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(3)
    n = 100
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(-5, 5, n)
    y = 2.0 * x1 - 0.5 * x2 + 3.0 + rng.normal(0, 0.5, n)
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y)


# ---------------------------------------------------------------------------
# MLPModel
# ---------------------------------------------------------------------------


def test_mlp_fit_predict_shape(linear_data):
    X, y = linear_data
    model = MLPModel(hidden_sizes=(8, 4), max_epochs=50, patience=10, seed=0).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (len(X),)


def test_mlp_is_reproducible_with_same_seed(linear_data):
    X, y = linear_data
    model_a = MLPModel(hidden_sizes=(8, 4), max_epochs=30, patience=10, seed=7).fit(X, y)
    model_b = MLPModel(hidden_sizes=(8, 4), max_epochs=30, patience=10, seed=7).fit(X, y)
    np.testing.assert_allclose(model_a.predict(X), model_b.predict(X))


def test_mlp_early_stopping_can_stop_before_max_epochs(linear_data):
    X, y = linear_data
    n = len(X)
    split = int(n * 0.8)
    model = MLPModel(hidden_sizes=(8, 4), max_epochs=1000, patience=5, seed=0).fit(
        X.iloc[:split], y.iloc[:split], X_val=X.iloc[split:], y_val=y.iloc[split:]
    )
    assert model.curve.epochs_run < 1000
    assert model.curve.best_epoch <= model.curve.epochs_run


def test_mlp_reports_parameter_count(linear_data):
    X, y = linear_data
    model = MLPModel(hidden_sizes=(8, 4), max_epochs=10, seed=0).fit(X, y)
    # 2 inputs -> 8 -> 4 -> 1, with biases: (2*8+8) + (8*4+4) + (4*1+1) = 24+36+5
    assert model.n_parameters() == (2 * 8 + 8) + (8 * 4 + 4) + (4 * 1 + 1)


def test_mlp_clips_to_output_range(linear_data):
    X, y = linear_data
    model = MLPModel(hidden_sizes=(4,), max_epochs=20, seed=0,
                     output_range=(0.0, 1.0)).fit(X, y)
    predictions = model.predict(X)
    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0


def test_mlp_handles_missing_values_via_median_imputation():
    rng = np.random.default_rng(1)
    n = 50
    X = pd.DataFrame({"x1": rng.uniform(0, 10, n)})
    X.loc[0, "x1"] = np.nan
    y = pd.Series(2.0 * X["x1"].fillna(X["x1"].median()) + 1.0)
    model = MLPModel(hidden_sizes=(4,), max_epochs=20, seed=0).fit(X, y)
    predictions = model.predict(X)
    assert np.all(np.isfinite(predictions))


# ---------------------------------------------------------------------------
# LSTMModel
# ---------------------------------------------------------------------------


@pytest.fixture
def sequence_data():
    rng = np.random.default_rng(5)
    n, seq_len, n_features = 60, 4, 2
    X = rng.normal(0, 1, (n, seq_len, n_features)).astype(np.float32)
    # Target depends on the last timestep only, learnable by a 1-layer LSTM.
    y = 2.0 * X[:, -1, 0] - X[:, -1, 1]
    return X, y


def test_lstm_fit_predict_shape(sequence_data):
    X, y = sequence_data
    model = LSTMModel(input_size=2, hidden_size=8, max_epochs=100, patience=20,
                      seed=0).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (len(X),)


def test_lstm_training_loss_decreases(sequence_data):
    X, y = sequence_data
    model = LSTMModel(input_size=2, hidden_size=8, max_epochs=100, patience=100,
                      seed=0).fit(X, y)
    assert model.curve.train_loss[-1] < model.curve.train_loss[0]


def test_lstm_reports_parameter_count(sequence_data):
    X, y = sequence_data
    model = LSTMModel(input_size=2, hidden_size=8, max_epochs=5, seed=0).fit(X, y)
    assert model.n_parameters() > 0


def test_set_seed_makes_torch_reproducible():
    set_seed(123)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_mlp_save_load_round_trip(linear_data, tmp_path):
    X, y = linear_data
    model = MLPModel(hidden_sizes=(8, 4), max_epochs=30, seed=0).fit(X, y)
    before = model.predict(X)

    path = model.save(tmp_path / "mlp.joblib")
    restored = MLPModel.load(path)
    after = restored.predict(X)

    np.testing.assert_allclose(before, after)
    assert restored.n_parameters() == model.n_parameters()


def test_lstm_save_load_round_trip(sequence_data, tmp_path):
    X, y = sequence_data
    model = LSTMModel(input_size=2, hidden_size=8, max_epochs=20, seed=0).fit(X, y)
    before = model.predict(X)

    path = model.save(tmp_path / "lstm.joblib")
    restored = LSTMModel.load(path)
    after = restored.predict(X)

    np.testing.assert_allclose(before, after)
