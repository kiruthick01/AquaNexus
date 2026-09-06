"""Training and benchmarking.

Trains the model set from ML_STRATEGY.md §6 across the split strategies in
:mod:`aquanexus.ml.splits` and tabulates the comparison §9.3 asks for.

Every run reports the mean predictor alongside the real models. On a grouped
dataset the test mean can drift from the training mean, so R² alone can look
respectable while the model adds nothing; the mean predictor makes the floor
explicit rather than implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.data.validator import validate_dataset
from aquanexus.logger import get_logger
from aquanexus.ml.evaluator import Metrics, compare, evaluate
from aquanexus.ml.models import HabitatPredictor, ModelConfig
from aquanexus.ml.splits import Split

log = get_logger("ml.trainer")

#: §6 model set, plus the constant-prediction floor.
DEFAULT_MODELS = ("xgboost", "random_forest", "linear", "mean")


@dataclass
class TrainingResult:
    """One fitted model and how it scored."""

    predictor: HabitatPredictor
    split_name: str
    validation: Metrics
    test: Metrics


def train_one(
    dataset: pd.DataFrame,
    features: list[str],
    split: Split,
    config: ModelConfig,
    target: str = "hsi",
) -> TrainingResult:
    """Fit one model on one split and score it."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split.apply(
        dataset, features, target
    )
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError(f"{split.name}: empty train or test partition")

    predictor = HabitatPredictor(config)
    predictor.fit(X_train, y_train,
                  validation=(X_val, y_val) if len(X_val) else None)

    baseline = float(np.mean(y_train))
    name = config.model_type
    val_metrics = (evaluate(y_val, predictor.predict(X_val), name,
                            f"{split.name}/val", baseline)
                   if len(X_val) else
                   evaluate(y_test, predictor.predict(X_test), name,
                            f"{split.name}/val", baseline))
    test_metrics = evaluate(y_test, predictor.predict(X_test), name,
                            split.name, baseline)

    log.info("%-14s %-10s RMSE %.4f  R2 %.3f  skill %.3f",
             name, split.name, test_metrics.rmse, test_metrics.r2,
             test_metrics.skill)
    return TrainingResult(predictor, split.name, val_metrics, test_metrics)


def benchmark(
    dataset: pd.DataFrame,
    features: list[str],
    splits: list[Split],
    models: tuple[str, ...] = DEFAULT_MODELS,
    target: str = "hsi",
    validate: bool = True,
) -> tuple[pd.DataFrame, dict[tuple[str, str], TrainingResult]]:
    """Train every model on every split.

    Returns the comparison table and the fitted models keyed by
    ``(split name, model type)``.
    """
    if validate:
        report = validate_dataset(dataset, features, target)
        if not report.ok:
            raise ValueError(f"dataset failed validation:\n{report}")
        for issue in report.warnings:
            log.warning("%s", issue)

    results: list[Metrics] = []
    fitted: dict[tuple[str, str], TrainingResult] = {}

    for split in splits:
        for model_type in models:
            config = ModelConfig(model_type=model_type)
            try:
                outcome = train_one(dataset, features, split, config, target)
            except Exception as exc:  # noqa: BLE001 - one bad model must not stop the sweep
                log.error("%s on %s failed: %s", model_type, split.name, exc)
                continue
            results.append(outcome.test)
            fitted[(split.name, model_type)] = outcome

    return compare(results), fitted


def select_best(fitted: dict[tuple[str, str], TrainingResult],
                split_name: str) -> TrainingResult | None:
    """Lowest test RMSE on one split, ignoring the constant-prediction floor."""
    candidates = [r for (s, m), r in fitted.items()
                  if s == split_name and m != "mean"]
    return min(candidates, key=lambda r: r.test.rmse) if candidates else None


def save_model(result: TrainingResult, path: Path | str | None = None) -> Path:
    path = Path(path) if path else settings.MODELS_DIR / "xgboost_v1.joblib"
    return result.predictor.save(path)
