"""Loads the trained models and prepares feature vectors for them.

Models are loaded once at startup and held in memory. Feature construction lives
here rather than in the route handlers, because both models need derived
quantities the caller does not supply - DO saturation from temperature, Froude
number from depth and velocity, cyclic month terms - and computing them in two
places would let them drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("api.registry")


@dataclass
class LoadedModel:
    """A model plus the metadata needed to answer honestly about it."""

    name: str
    predictor: object
    unit: str
    labels: str
    model_type: str
    n_train: int
    features: list[str]
    metrics: dict = field(default_factory=dict)
    caveats: list[str] = field(default_factory=list)
    training_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


class ModelRegistry:
    """Holds the loaded models and builds their inputs."""

    def __init__(self) -> None:
        self.models: dict[str, LoadedModel] = {}
        self.manifest: dict = {}
        self.error: str | None = None

    # -- loading ------------------------------------------------------------

    def load(self, models_dir: Path | None = None) -> ModelRegistry:
        from aquanexus.ml.models import HabitatPredictor

        models_dir = models_dir or settings.MODELS_DIR
        manifest_path = models_dir / "manifest.json"
        if not manifest_path.is_file():
            self.error = (f"no manifest at {manifest_path}; "
                          "run scripts/train_models.py")
            log.error(self.error)
            return self

        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, meta in self.manifest.get("models", {}).items():
            path = models_dir / meta["path"]
            if not path.is_file():
                log.error("%s: %s is missing", name, path.name)
                continue
            try:
                predictor = HabitatPredictor.load(path)
            except Exception as exc:  # noqa: BLE001 - one bad model must not stop the rest
                log.error("%s failed to load: %s", name, exc)
                continue

            self.models[name] = LoadedModel(
                name=name, predictor=predictor, unit=meta.get("unit", ""),
                labels=meta.get("labels", "observed"),
                model_type=meta.get("model_type", "unknown"),
                n_train=int(meta.get("n_train", 0)),
                features=list(meta.get("features", predictor.feature_names)),
                metrics=meta.get("metrics", {}),
                caveats=list(meta.get("caveats", [])),
                training_ranges={k: tuple(v) for k, v in
                                 meta.get("training_ranges", {}).items()},
            )
            log.info("loaded %s (%s, %s labels)", name, meta.get("model_type"),
                     meta.get("labels"))

        if not self.models:
            self.error = "no models could be loaded"
        return self

    def get(self, name: str) -> LoadedModel:
        if name not in self.models:
            raise KeyError(f"model {name!r} is not loaded; "
                           f"available: {sorted(self.models)}")
        return self.models[name]

    @property
    def ready(self) -> bool:
        return bool(self.models)

    # -- feature construction ------------------------------------------------

    def build_features(self, state: dict, model: LoadedModel) -> pd.DataFrame:
        """Turn a caller's environmental state into the model's feature row.

        Derived quantities are computed here. Anything the caller did not supply
        and that cannot be derived is left as NaN - the tree models split on it
        natively and the linear pipeline imputes it, so a partial observation
        still gets an answer rather than a validation error.
        """
        from aquanexus.data.preprocessor import do_saturation, froude_number

        values = dict(state)

        temperature = values.get("water_temp")
        if temperature is not None:
            values.setdefault("do_saturation", float(do_saturation(temperature)))

        depth, velocity = values.get("depth"), values.get("velocity")
        if depth is not None and velocity is not None and depth > 0:
            values.setdefault("froude_number", float(froude_number(velocity, depth)))
            values.setdefault("reach_froude", values["froude_number"])
        # The reach_* features are reach means in training; a caller supplying a
        # single point is treated as describing that reach.
        for source, alias in (("depth", "reach_depth"), ("velocity", "reach_velocity")):
            if values.get(source) is not None:
                values.setdefault(alias, values[source])

        month = values.get("month")
        if month is not None:
            values["month_sin"] = float(np.sin(2 * np.pi * month / 12.0))
            values["month_cos"] = float(np.cos(2 * np.pi * month / 12.0))

        if values.get("dissolved_oxygen") is not None and "do_saturation" in values:
            values.setdefault("do_deficit",
                              values["do_saturation"] - values["dissolved_oxygen"])

        row = {name: values.get(name, np.nan) for name in model.features}
        return pd.DataFrame([row], columns=model.features)

    def check_ranges(self, state: dict, model: LoadedModel) -> list[dict]:
        """Report values outside the model's training range.

        Reported, not rejected. A model asked about conditions it has never seen
        should answer and say so, rather than refuse or pretend.
        """
        warnings = []
        for name, (low, high) in model.training_ranges.items():
            value = state.get(name)
            if value is None or not np.isfinite(value):
                continue
            if value < low or value > high:
                warnings.append({"feature": name, "value": float(value),
                                 "training_min": float(low),
                                 "training_max": float(high)})
        return warnings


#: Process-wide registry, populated on application startup.
registry = ModelRegistry()
