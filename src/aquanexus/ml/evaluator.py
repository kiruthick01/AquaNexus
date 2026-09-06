"""Metrics and model comparison.

Reports the metrics ML_STRATEGY.md §9 asks for, plus two it omits that change how
the others should be read:

* **Skill against the mean.** R² is measured against predicting the training
  mean, but on a grouped dataset the test mean can differ from the training mean
  enough for R² to look respectable while the model adds almost nothing.
  ``skill`` states the improvement over that floor directly.
* **Nash-Sutcliffe efficiency**, standard in hydrology and requested by §9.1,
  which is R² computed against the *test* mean rather than the fitted line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("ml.evaluator")


@dataclass
class Metrics:
    """Regression metrics for one model on one split."""

    model: str
    split: str
    n: int
    rmse: float
    mae: float
    r2: float
    pearson: float
    nse: float
    bias: float
    skill: float = float("nan")
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"model": self.model, "split": self.split, "n": self.n,
                "rmse": self.rmse, "mae": self.mae, "r2": self.r2,
                "pearson": self.pearson, "nse": self.nse, "bias": self.bias,
                "skill": self.skill, **self.extra}


def evaluate(y_true, y_pred, model: str = "", split: str = "",
             baseline: float | None = None) -> Metrics:
    """Compute metrics for one set of predictions.

    ``baseline`` is the constant a naive predictor would emit - normally the
    training mean. Given it, ``skill`` reports the fractional reduction in mean
    squared error against that constant: 0 means no better than guessing the
    mean, 1 means perfect.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]
    if len(y_true) == 0:
        raise ValueError("no finite pairs to evaluate")

    residual = y_pred - y_true
    mse = float(np.mean(residual**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residual)))
    bias = float(np.mean(residual))

    variance = float(np.mean((y_true - y_true.mean()) ** 2))
    nse = 1.0 - mse / variance if variance > 0 else float("nan")

    if y_pred.std() > 0 and y_true.std() > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")

    skill = float("nan")
    if baseline is not None:
        base_mse = float(np.mean((y_true - baseline) ** 2))
        skill = 1.0 - mse / base_mse if base_mse > 0 else float("nan")

    return Metrics(model=model, split=split, n=len(y_true), rmse=rmse, mae=mae,
                   r2=nse, pearson=pearson, nse=nse, bias=bias, skill=skill)


def compare(results: list[Metrics]) -> pd.DataFrame:
    """Tabulate metrics, best RMSE first within each split."""
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame([m.as_row() for m in results])
    return frame.sort_values(["split", "rmse"]).reset_index(drop=True)


def residual_summary(y_true, y_pred, groups=None) -> pd.DataFrame:
    """Where the errors sit, by decile of the true value.

    A model can post a good overall RMSE while being systematically wrong at the
    extremes - which for a habitat index is exactly where the answer matters,
    since those are the degraded and the pristine sites.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=float),
                          "pred": np.asarray(y_pred, dtype=float)})
    frame["residual"] = frame["pred"] - frame["y"]
    frame["decile"] = pd.qcut(frame["y"], 10, labels=False, duplicates="drop")
    if groups is not None:
        frame["group"] = np.asarray(groups)

    summary = frame.groupby("decile").agg(
        n=("y", "size"),
        y_mean=("y", "mean"),
        pred_mean=("pred", "mean"),
        bias=("residual", "mean"),
        rmse=("residual", lambda r: float(np.sqrt(np.mean(r**2)))),
    )
    return summary.round(4)
