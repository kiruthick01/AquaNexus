"""Model validation and baseline comparison (Phase 2c).

Implements the validation and baselines AQUANEXUS-PLAN.md Phase 2c and
ML_STRATEGY.md §7.3 ask for, adapted where the spec assumes data that does not
exist.

The three specified baselines
-----------------------------
1. **Hydraulic-only** - depth, velocity, flow alone, showing what the chemistry
   and thermal features add.
2. **Linear regression** - specified to show the value of non-linear ML. On the
   observed-DO target it is the *best* model rather than a foil, which is
   reported as found.
3. **Persistence** - the spec says "previous day's score". Sampling is monthly,
   so this is the previous *sample* at the same station. That is the honest
   version and a genuinely strong baseline for a slowly varying quantity.

Persistence is the baseline that matters most and the one most often omitted. A
model that cannot beat "same as last time" has not earned its complexity,
whatever its R².
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger
from aquanexus.ml.evaluator import Metrics, compare, evaluate
from aquanexus.ml.models import HabitatPredictor, ModelConfig

log = get_logger("ml.validator")


@dataclass
class ValidationReport:
    """Results of one validation campaign."""

    target: str
    n: int
    results: list[Metrics] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        return compare(self.results)

    def best(self, exclude: tuple[str, ...] = ("mean", "persistence")) -> Metrics | None:
        candidates = [m for m in self.results if m.model not in exclude]
        return min(candidates, key=lambda m: m.rmse) if candidates else None

    def __str__(self) -> str:
        lines = [f"{self.target}: n={self.n}", self.table().round(4).to_string(index=False)]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def persistence_baseline(
    frame: pd.DataFrame,
    target: str,
    group_column: str = "station",
    time_column: str = "timestamp",
) -> pd.Series:
    """Predict each observation as the previous observation at the same station.

    The spec's "previous day" becomes "previous sample", since sampling is
    monthly. The first sample at each station has no predecessor and is left NaN
    rather than filled - back-filling it would score the baseline on a value it
    could not have known.
    """
    ordered = frame.sort_values([group_column, time_column])
    lagged = ordered.groupby(group_column, sort=False)[target].shift(1)
    return lagged.reindex(frame.index)


def hydraulic_only_features(features: list[str]) -> list[str]:
    """The hydraulic subset, for the 'what do the other features add' baseline."""
    return [f for f in features
            if f.startswith("reach_") or f in {"depth", "velocity", "discharge",
                                               "flow_area", "top_width",
                                               "froude_number", "shear_stress"}]


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class ModelValidator:
    """Runs a model and its baselines through grouped cross-validation."""

    def __init__(self, frame: pd.DataFrame, features: list[str], target: str,
                 group_column: str = "station"):
        self.frame = frame.reset_index(drop=True)
        self.features = [f for f in features if f in self.frame.columns]
        self.target = target
        self.group_column = group_column

        missing = set(features) - set(self.features)
        if missing:
            log.warning("features absent and skipped: %s", sorted(missing))

    # -- cross-validated prediction -----------------------------------------

    def cross_val_predict(self, config: ModelConfig) -> np.ndarray:
        """Out-of-fold predictions with whole groups held out.

        Grouping is the point: with a handful of stations and monthly sampling,
        an ungrouped fold puts the same station either side of the split and
        reports how well the model recognises a station rather than how well it
        predicts one it has not seen.
        """
        from sklearn.model_selection import GroupKFold

        X = self.frame[self.features]
        y = self.frame[self.target]
        groups = self.frame[self.group_column]
        n_splits = min(groups.nunique(), len(self.frame))
        if n_splits < 2:
            raise ValueError("need at least two groups to cross-validate")

        predictions = np.full(len(self.frame), np.nan)
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
            model = HabitatPredictor(config).fit(X.iloc[train_idx], y.iloc[train_idx])
            predictions[test_idx] = model.predict(X.iloc[test_idx])
        return predictions

    # -- the campaign --------------------------------------------------------

    def run(self, model_types: tuple[str, ...] = ("xgboost", "random_forest", "linear"),
            output_range: tuple[float, float] | None = None) -> ValidationReport:
        """Evaluate every model plus the three specified baselines."""
        y = self.frame[self.target]
        baseline_mean = float(y.mean())
        report = ValidationReport(target=self.target, n=len(self.frame))

        for model_type in model_types:
            config = ModelConfig(model_type=model_type, max_depth=3,
                                 n_estimators=150, output_range=output_range)
            try:
                predictions = self.cross_val_predict(config)
                report.results.append(
                    evaluate(y, predictions, model_type, "grouped-CV", baseline_mean)
                )
            except Exception as exc:  # noqa: BLE001 - one model must not stop the sweep
                log.error("%s failed: %s", model_type, exc)

        # Baseline 1: hydraulics alone.
        hydraulic = hydraulic_only_features(self.features)
        if hydraulic and len(hydraulic) < len(self.features):
            sub = ModelValidator(self.frame, hydraulic, self.target, self.group_column)
            config = ModelConfig(model_type="linear", output_range=output_range)
            report.results.append(
                evaluate(y, sub.cross_val_predict(config),
                         "hydraulic-only", "grouped-CV", baseline_mean)
            )
            report.notes.append(
                f"hydraulic-only uses {len(hydraulic)} of {len(self.features)} features"
            )

        # Baseline 2: persistence.
        lagged = persistence_baseline(self.frame, self.target, self.group_column)
        usable = lagged.notna()
        if usable.sum() >= 10:
            report.results.append(
                evaluate(y[usable], lagged[usable], "persistence", "grouped-CV",
                         baseline_mean)
            )
            report.notes.append(
                f"persistence scored on {int(usable.sum())} of {len(self.frame)} rows "
                "(first sample per station has no predecessor)"
            )

        # Baseline 3: the constant floor.
        report.results.append(
            evaluate(y, np.full(len(y), baseline_mean), "mean", "grouped-CV",
                     baseline_mean)
        )

        log.info("%s: %d model(s)/baseline(s) evaluated", self.target,
                 len(report.results))
        return report

    # -- generalisation checks ----------------------------------------------

    def by_group(self, config: ModelConfig) -> pd.DataFrame:
        """Per-group scores, so one badly-predicted station cannot hide in a mean.

        This is the spec's spatial validation in the form the data supports:
        every station is held out in turn and scored on its own.
        """
        predictions = self.cross_val_predict(config)
        y = self.frame[self.target]
        rows = []
        for name, index in self.frame.groupby(self.group_column).groups.items():
            mask = self.frame.index.isin(index)
            actual, predicted = y[mask], predictions[mask]
            finite = np.isfinite(predicted)
            if finite.sum() < 2:
                continue
            residual = predicted[finite] - actual[finite]
            rows.append({
                self.group_column: name,
                "n": int(finite.sum()),
                "observed_mean": float(actual[finite].mean()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "bias": float(np.mean(residual)),
            })
        return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)

    def by_condition(self, config: ModelConfig, column: str,
                     quantile: float = 0.85) -> pd.DataFrame:
        """Scores inside and outside an extreme of ``column``.

        The spec's event-based validation, reframed: rather than named flood and
        drought events, split on the tail of a driver and check the model does
        not fall apart there.
        """
        predictions = self.cross_val_predict(config)
        y = self.frame[self.target]
        values = self.frame[column]
        high, low = values.quantile(quantile), values.quantile(1 - quantile)

        rows = []
        for label, mask in (("low tail", values <= low),
                            ("middle", (values > low) & (values < high)),
                            ("high tail", values >= high)):
            actual, predicted = y[mask], predictions[mask]
            finite = np.isfinite(predicted)
            if finite.sum() < 2:
                continue
            residual = predicted[finite] - actual[finite]
            rows.append({
                "band": label, "n": int(finite.sum()),
                f"{column}_mean": float(values[mask][finite].mean()),
                "rmse": float(np.sqrt(np.mean(residual**2))),
                "bias": float(np.mean(residual)),
            })
        return pd.DataFrame(rows)
