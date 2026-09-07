"""SHAP explanations, interaction analysis and threshold discovery.

Implements ML_STRATEGY.md §7.2 (Phase 2b) and the explainer interface sketched in
AQUANEXUS-PLAN.md Phase 2b.

Deviations from the spec, each forced by what the data turned out to be:

* **§7.2 Step 2 specifies ``TreeExplainer``** because it assumes XGBoost is the
  chosen model. On the observed dissolved-oxygen target Ridge cross-validates
  better than either tree model (see ``docs/ML_METHODOLOGY.md``), so the
  explainer dispatches on model type: ``TreeExplainer`` for tree ensembles,
  ``LinearExplainer`` for linear models, ``KernelExplainer`` as the fallback.
  Hard-coding TreeExplainer would simply fail on the model we actually ship.
* **Step 2 also specifies a background set of 8,760 records** and **Step 3 a test
  set of 8,760, sampled to 1,000 if slow.** The real datasets are 138 rows (DO)
  and 7,314 (synthetic HSI). Background is taken from the training partition and
  summarised with k-means when large, which is what SHAP recommends and what
  keeps KernelExplainer tractable.

Threshold discovery is not a SHAP feature - it is implemented here by sweeping one
variable across its observed range with everything else held at its median, which
gives the marginal response curve the spec's "value where prediction changes
significantly" asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("ml.explainer")


def collinear_pairs(frame: pd.DataFrame, features: list[str] | None = None,
                    threshold: float = 0.9) -> pd.DataFrame:
    """Feature pairs correlated beyond ``threshold``, strongest first.

    Kept as a plain function so it can be computed at training time - the API
    serves the pairs alongside every explanation, and a caller who cannot see
    that depth, velocity and top width are near-duplicates will read their
    separate SHAP contributions as three independent findings.
    """
    columns = [f for f in (features or frame.columns) if f in frame.columns]
    correlations = frame[columns].corr(numeric_only=True)

    pairs = []
    names = list(correlations.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            value = correlations.loc[a, b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append({"feature_a": a, "feature_b": b,
                              "correlation": float(value)})

    if not pairs:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])
    return (pd.DataFrame(pairs)
            .sort_values("correlation", key=abs, ascending=False)
            .reset_index(drop=True))


@dataclass
class Explanation:
    """SHAP attribution for a single prediction."""

    prediction: float
    expected_value: float
    contributions: pd.Series
    top_features: list[tuple[str, float]] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"prediction {self.prediction:.3f} "
                 f"(baseline {self.expected_value:.3f})"]
        for name, value in self.top_features:
            lines.append(f"   {name:22s} {value:+.4f}")
        return "\n".join(lines)


class HabitatExplainer:
    """SHAP wrapper around a fitted :class:`~aquanexus.ml.models.HabitatPredictor`."""

    def __init__(self, predictor):
        if not getattr(predictor, "is_fitted", False):
            raise RuntimeError("model must be fitted before it can be explained")
        self.predictor = predictor
        self.feature_names: list[str] = list(predictor.feature_names)
        self.explainer: Any = None
        self.shap_values: np.ndarray | None = None
        self._explained: pd.DataFrame | None = None

    # -- setup --------------------------------------------------------------

    def fit_explainer(self, background: pd.DataFrame, max_background: int = 100):
        """Build the SHAP explainer appropriate to the wrapped model.

        ``background`` should come from the **training** partition. Using the
        rows being explained would make the reference distribution depend on the
        thing under test.
        """
        import shap

        rows = background[self.feature_names]
        model = self.predictor.model
        model_type = self.predictor.config.model_type

        if model_type in {"xgboost", "random_forest"}:
            self.explainer = shap.TreeExplainer(model)
            log.info("TreeExplainer for %s", model_type)
        elif model_type == "linear":
            # The linear model is a pipeline (impute -> scale -> Ridge). SHAP
            # cannot see through it, so the whole pipeline is explained through
            # the model-agnostic path against a summarised background.
            summary = self._summarise(rows, max_background)
            self.explainer = shap.KernelExplainer(
                lambda data: model.predict(pd.DataFrame(data, columns=self.feature_names)),
                summary,
            )
            # shap.kmeans returns DenseData, which has no len().
            size = getattr(summary, "data", summary)
            log.info("KernelExplainer for the linear pipeline (%d background rows)",
                     len(size))
        else:
            summary = self._summarise(rows, max_background)
            self.explainer = shap.KernelExplainer(
                lambda data: self.predictor.predict(
                    pd.DataFrame(data, columns=self.feature_names)), summary
            )
            log.info("KernelExplainer for %s", model_type)
        return self

    @staticmethod
    def _summarise(rows: pd.DataFrame, limit: int):
        """k-means summary of the background, as SHAP recommends for large sets."""
        import shap

        if len(rows) <= limit:
            return rows
        return shap.kmeans(rows, limit)

    # -- values -------------------------------------------------------------

    def compute(self, X: pd.DataFrame, max_samples: int | None = None) -> np.ndarray:  # noqa: N803
        """Compute SHAP values for the given rows."""
        if self.explainer is None:
            raise RuntimeError("call fit_explainer() first")

        rows = X[self.feature_names]
        if max_samples and len(rows) > max_samples:
            rows = rows.sample(max_samples, random_state=42)
            log.info("sampled %d of %d rows", max_samples, len(X))

        values = self.explainer.shap_values(rows)
        if isinstance(values, list):  # multi-output
            values = values[0]
        self.shap_values = np.asarray(values)
        self._explained = rows
        log.info("SHAP values: %s", self.shap_values.shape)
        return self.shap_values

    @property
    def expected_value(self) -> float:
        value = getattr(self.explainer, "expected_value", 0.0)
        return float(np.ravel(value)[0])

    # -- §7.2 Step 5: importance ranking -------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        """Rank features by mean |SHAP|, with the signed mean alongside.

        The signed mean matters: two features can move predictions equally hard
        while one consistently raises them and the other consistently lowers
        them, and the magnitude ranking alone hides that.

        **Collinear features share credit arbitrarily.** The hydraulic features
        here are all derived from discharge through the same model, so depth,
        velocity, top width and Froude number move together. SHAP still sums
        correctly to the prediction, but how it divides the total between such
        features is not a statement about which one matters - swapping which is
        listed first can change the ranking without changing the model. Use
        :meth:`collinearity` before reading much into individual ranks.
        """
        self._require_values()
        magnitude = np.abs(self.shap_values).mean(axis=0)
        signed = self.shap_values.mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names,
                          "mean_abs_shap": magnitude,
                          "mean_shap": signed})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def collinearity(self, threshold: float = 0.9) -> pd.DataFrame:
        """Feature pairs correlated beyond ``threshold`` in the explained rows.

        Any pair listed here shares SHAP credit arbitrarily, so their individual
        importances should be read as one combined contribution.

        The count depends on **which rows are explained**: a single station's
        subset is not the whole record, and pairs move in and out of the list
        with it. Explain the full training set before quoting a number.
        """
        self._require_values()
        return collinear_pairs(self._explained, threshold=threshold)

    # -- single prediction ---------------------------------------------------

    def explain_prediction(self, row: pd.DataFrame, top: int = 5) -> Explanation:
        """Explain one row: contribution of each feature, largest first."""
        if self.explainer is None:
            raise RuntimeError("call fit_explainer() first")

        single = row[self.feature_names].iloc[[0]] if len(row) > 1 else row[self.feature_names]
        values = self.explainer.shap_values(single)
        if isinstance(values, list):
            values = values[0]
        contributions = pd.Series(np.ravel(values), index=self.feature_names)
        ordered = contributions.reindex(contributions.abs().sort_values(ascending=False).index)

        return Explanation(
            prediction=float(self.predictor.predict(single)[0]),
            expected_value=self.expected_value,
            contributions=contributions,
            top_features=[(k, float(v)) for k, v in ordered.head(top).items()],
        )

    # -- §7.2 Step 6: interactions ------------------------------------------

    def interaction_analysis(self, feature_a: str, feature_b: str,
                             quantiles: int = 2) -> dict:
        """Test whether two features interact, by comparing the joint effect
        against the sum of the individual effects.

        Splits the explained rows into high/low bins on each feature and compares
        the mean prediction in the both-adverse corner against what the two
        one-at-a-time changes predict additively. A negative ``interaction``
        means the combination is worse than the parts - synergistic stress, the
        case ML_STRATEGY §3.2 cares about for temperature and oxygen.
        """
        self._require_values()
        rows = self._explained
        for name in (feature_a, feature_b):
            if name not in rows.columns:
                raise KeyError(f"{name!r} is not among the explained features")

        predictions = pd.Series(self.predictor.predict(rows), index=rows.index)
        a_high = rows[feature_a] > rows[feature_a].median()
        b_high = rows[feature_b] > rows[feature_b].median()

        masks = {
            "low_low": ~a_high & ~b_high,
            "high_low": a_high & ~b_high,
            "low_high": ~a_high & b_high,
            "high_high": a_high & b_high,
        }
        counts = {name: int(mask.sum()) for name, mask in masks.items()}
        cells = {name: float(predictions[mask].mean()) if counts[name] else float("nan")
                 for name, mask in masks.items()}

        # An empty corner means the two features are collinear enough that one
        # combination never occurs - water temperature and DO saturation, for
        # instance, are deterministically related, so "both high" cannot happen.
        # The design is unidentifiable, and returning a bare NaN invites reading
        # it as "no interaction".
        empty = [name for name, count in counts.items() if count == 0]
        correlation = float(rows[feature_a].corr(rows[feature_b]))

        result = {"feature_a": feature_a, "feature_b": feature_b, **cells,
                  "counts": counts, "correlation": correlation,
                  "n": int(len(rows)), "identifiable": not empty}

        if empty:
            result.update(additive_effect=float("nan"),
                          observed_effect=float("nan"),
                          interaction=float("nan"),
                          reason=f"empty cell(s) {empty}; features correlate "
                                 f"{correlation:+.2f} so the design is "
                                 f"unidentifiable")
            log.warning("%s x %s: not identifiable - %s",
                        feature_a, feature_b, result["reason"])
            return result

        additive = ((cells["high_low"] - cells["low_low"])
                    + (cells["low_high"] - cells["low_low"]))
        observed = cells["high_high"] - cells["low_low"]
        result.update(additive_effect=float(additive),
                      observed_effect=float(observed),
                      interaction=float(observed - additive))

        if abs(correlation) > 0.9:
            result["warning"] = (f"features correlate {correlation:+.2f}; the "
                                 "split between them is not meaningful")
            log.warning("%s x %s: %s", feature_a, feature_b, result["warning"])

        log.info("%s x %s: interaction %+.4f", feature_a, feature_b,
                 result["interaction"])
        return result

    # -- threshold discovery -------------------------------------------------

    def threshold_discovery(self, data: pd.DataFrame, feature: str,
                            steps: int = 40, sensitivity: float = 0.5) -> dict:
        """Marginal response of the prediction to one feature.

        Sweeps ``feature`` across its observed 1st-99th percentile range with
        every other feature held at its median, and reports where the response
        turns over most sharply.

        This is a **partial-dependence style** curve: it shows what the model
        does, not what the river does. Correlated features are held fixed at
        values that may not co-occur with the swept value in reality.
        """
        if feature not in self.feature_names:
            raise KeyError(f"{feature!r} is not a model feature")

        rows = data[self.feature_names]
        low, high = rows[feature].quantile([0.01, 0.99])
        if not np.isfinite([low, high]).all() or low == high:
            raise ValueError(f"{feature!r} has no usable range to sweep")

        grid = np.linspace(low, high, steps)
        template = rows.median(numeric_only=True)
        sweep = pd.DataFrame([template] * steps, columns=self.feature_names)
        sweep[feature] = grid

        predictions = self.predictor.predict(sweep)
        gradient = np.gradient(predictions, grid)

        sharpest = int(np.argmax(np.abs(gradient)))
        span = float(predictions.max() - predictions.min())
        return {
            "feature": feature,
            "grid": grid,
            "prediction": predictions,
            "gradient": gradient,
            "threshold": float(grid[sharpest]),
            "max_gradient": float(gradient[sharpest]),
            "response_span": span,
            # A feature whose sweep barely moves the prediction has no threshold
            # worth quoting, however sharp its steepest point looks.
            "is_influential": bool(span > sensitivity * float(np.std(
                self.predictor.predict(rows)))),
        }

    # -- plots ---------------------------------------------------------------

    def plot_summary(self, save_path: Path | str | None = None, kind: str = "bar"):
        """SHAP summary plot. Returns the matplotlib figure."""
        import matplotlib.pyplot as plt
        import shap

        self._require_values()
        plt.figure()
        shap.summary_plot(self.shap_values, self._explained,
                          plot_type=kind, show=False)
        figure = plt.gcf()
        if save_path:
            self._save(figure, save_path)
        return figure

    def plot_dependence(self, feature: str, save_path: Path | str | None = None,
                        interaction: str | None = None):
        """SHAP dependence plot for one feature."""
        import matplotlib.pyplot as plt
        import shap

        self._require_values()
        plt.figure()
        shap.dependence_plot(feature, self.shap_values, self._explained,
                             interaction_index=interaction, show=False)
        figure = plt.gcf()
        if save_path:
            self._save(figure, save_path)
        return figure

    @staticmethod
    def _save(figure, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=140, bbox_inches="tight")
        log.info("wrote %s", path)
        return path

    # -- persistence ---------------------------------------------------------

    def save_values(self, path: Path | str) -> Path:
        """Save SHAP values, as §7.2 Step 3 asks."""
        self._require_values()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.shap_values)
        return path

    def _require_values(self) -> None:
        if self.shap_values is None:
            raise RuntimeError("call compute() first")
