"""Generic save/load for the model objects across ``ml/*.py``.

One `joblib.dump`/`load` pair, reused everywhere a Phase 2+ model needs
persistence, rather than a near-identical implementation in every module.
Suitable for any object made of picklable pieces - numpy arrays, sklearn
pipelines/estimators, PyTorch modules and tensors - which covers every model
in this project except the PyMC ones (`ml/bayesian.py` implements its own
save/load, since a fitted `pymc.Model`/`arviz.InferenceData` pair is not
reliably picklable and carries far more than a predictor needs to keep).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def save_model(obj: object, path: Path | str) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_model(cls: type[T], path: Path | str) -> T:
    import joblib

    obj = joblib.load(Path(path))
    if not isinstance(obj, cls):
        raise TypeError(f"{path} does not contain a {cls.__name__} (got {type(obj).__name__})")
    return obj
