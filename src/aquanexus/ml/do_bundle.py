"""A fitted dissolved-oxygen model packaged with everything needed to serve it.

A bundle is a directory holding the LightGBM booster (``model.txt``) and
``bundle.json``: the feature list, the conformal interval widths per predicted-DO
bin, the held-out evaluation numbers the model was reported with, and how it was
trained. Keeping these together means a served prediction can always say which
data it was fitted on and how far to trust it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

BOOSTER_FILE = "model.txt"
CLASSIFIER_FILE = "classifier.txt"
META_FILE = "bundle.json"


@dataclass
class DOBundle:
    booster: lgb.Booster
    features: list[str]
    bin_edges: list[float]
    bin_widths: list[float]
    level: float
    metadata: dict = field(default_factory=dict)
    #: Optional model of P(DO < 5 mg/L), with an isotonic map (``x`` -> ``y``) that
    #: recalibrates its raw output on stations it was not fitted on.
    classifier: lgb.Booster | None = None
    calibration: dict | None = None

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Point prediction (mg/L) and a split-conformal interval for each row."""
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise KeyError(f"missing features: {missing}")
        pred = self.booster.predict(frame[self.features], num_iteration=self.booster.best_iteration)
        half = np.asarray(self.bin_widths)[np.digitize(pred, self.bin_edges)]
        out = pd.DataFrame(
            {"prediction": pred, "lower": pred - half, "upper": pred + half}, index=frame.index
        )
        if self.classifier is not None:
            raw = self.classifier.predict(
                frame[self.features], num_iteration=self.classifier.best_iteration
            )
            if self.calibration:
                raw = np.interp(raw, self.calibration["x"], self.calibration["y"])
            out["p_low"] = raw
        return out

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(
            str(directory / BOOSTER_FILE), num_iteration=self.booster.best_iteration
        )
        meta = {
            "features": self.features,
            "bin_edges": self.bin_edges,
            "bin_widths": self.bin_widths,
            "level": self.level,
            "metadata": self.metadata,
            "calibration": self.calibration,
        }
        if self.classifier is not None:
            self.classifier.save_model(
                str(directory / CLASSIFIER_FILE), num_iteration=self.classifier.best_iteration
            )
        (directory / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False), "utf-8")
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> DOBundle:
        directory = Path(directory)
        meta = json.loads((directory / META_FILE).read_text("utf-8"))
        booster = lgb.Booster(model_file=str(directory / BOOSTER_FILE))
        classifier_path = directory / CLASSIFIER_FILE
        classifier = (
            lgb.Booster(model_file=str(classifier_path)) if classifier_path.exists() else None
        )
        return cls(
            booster=booster,
            features=meta["features"],
            bin_edges=meta["bin_edges"],
            bin_widths=meta["bin_widths"],
            level=meta["level"],
            metadata=meta["metadata"],
            classifier=classifier,
            calibration=meta.get("calibration"),
        )
