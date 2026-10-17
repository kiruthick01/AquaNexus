"""Splits, baselines, LightGBM and tuning for the nationwide DO model.

Four evaluation regimes, each answering a different question:

``temporal``
    Train FY2013-2019, test FY2021-2022 (2020 is a purge year). Same stations,
    later years: does the model hold up as the climate and the rivers drift?
``station``
    A fifth of stations are never seen in training, all years. Does it work at a
    river it has not been shown?
``strict``
    Unseen stations *and* later years. The hardest and most honest regime, and the
    headline number.
``prefecture``
    Five-fold, whole prefectures held out. Does it cross regional boundaries?

Nothing a test row could influence is ever used to fit anything: baselines are
fitted on the training mask only, tuning cross-validates by station inside the
training mask, and early stopping uses stations carved out of the training mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aquanexus.data.features import TARGET
from aquanexus.logger import get_logger

log = get_logger("ml.do_experiment")

LOW_DO = 5.0  # mg/L, below which fish are stressed
TRAIN_LAST_FY, TEST_FIRST_FY = 2019, 2021


@dataclass(frozen=True)
class Split:
    name: str
    train: np.ndarray
    test: np.ndarray

    def __post_init__(self):
        if (self.train & self.test).any():
            raise ValueError(f"split {self.name!r}: train and test overlap")


def make_splits(frame: pd.DataFrame, seed: int = 0, station_frac: float = 0.2) -> list[Split]:
    """The three single-shot regimes. Prefecture folds come from :func:`prefecture_folds`."""
    rng = np.random.default_rng(seed)
    stations = np.sort(frame["station_id"].unique())
    held = set(rng.choice(stations, size=int(len(stations) * station_frac), replace=False))
    is_held = frame["station_id"].isin(held).to_numpy()
    early = (frame["fiscal_year"] <= TRAIN_LAST_FY).to_numpy()
    late = (frame["fiscal_year"] >= TEST_FIRST_FY).to_numpy()
    return [
        Split("temporal", early, late),
        Split("station", ~is_held, is_held),
        Split("strict", early & ~is_held, late & is_held),
    ]


def prefecture_folds(frame: pd.DataFrame, n_splits: int = 5) -> list[Split]:
    groups = frame["pref_code"].to_numpy()
    folds = []
    for i, (train, test) in enumerate(GroupKFold(n_splits).split(frame, groups=groups)):
        mask = lambda idx: np.isin(np.arange(len(frame)), idx)  # noqa: E731
        folds.append(Split(f"prefecture_{i}", mask(train), mask(test)))
    return folds


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """R2, MAE, RMSE, bias, and how well the fish-stress tail (DO < 5 mg/L) is found."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = ~np.isnan(pred)
    y, pred = y[ok], pred[ok]
    err = pred - y
    ss_tot = ((y - y.mean()) ** 2).sum()
    low = y < LOW_DO
    flagged = pred < LOW_DO + 1.0  # alarm one mg/L early: missing a sick river is the costly error
    return {
        "n": int(len(y)),
        "r2": float(1 - (err**2).sum() / ss_tot),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "n_low": int(low.sum()),
        "mae_low": float(np.abs(err[low]).mean()) if low.any() else float("nan"),
        "bias_low": float(err[low].mean()) if low.any() else float("nan"),
        "recall_low": float(flagged[low].mean()) if low.any() else float("nan"),
        "precision_low": float(low[flagged].mean()) if flagged.any() else float("nan"),
    }


# --- baselines ---------------------------------------------------------------------


def global_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.full(len(test), train[TARGET].mean())


def climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Mean DO by station-month, falling back to station, prefecture-month, then global.

    The bar every learned model has to clear: it knows each station's seasonal cycle
    and nothing about the weather. Fitted on ``train`` only.
    """
    keys = [
        (["station_id", "month"], "a"),
        (["station_id"], "b"),
        (["pref_code", "month"], "c"),
    ]
    pred = pd.Series(np.nan, index=test.index)
    for cols, tag in keys:
        table = train.groupby(cols)[TARGET].mean().rename(tag).reset_index()
        looked = test[cols].merge(table, on=cols, how="left")[tag]
        looked.index = test.index
        pred = pred.fillna(looked)
    return pred.fillna(train[TARGET].mean()).to_numpy()


def persistence(test: pd.DataFrame) -> np.ndarray:
    """Last visit's DO at the same station. NaN on a station's first visit."""
    return test["prev_do"].to_numpy()


def ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[features], train[TARGET])
    return model.predict(test[features])


# --- LightGBM ----------------------------------------------------------------------

DEFAULT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "n_jobs": -1,
}


def _carve_validation(train: pd.DataFrame, frac: float = 0.1, seed: int = 0):
    """Hold out whole stations for early stopping so it cannot see a station it trains on."""
    rng = np.random.default_rng(seed)
    stations = np.sort(train["station_id"].unique())
    val = set(rng.choice(stations, size=max(1, int(len(stations) * frac)), replace=False))
    mask = train["station_id"].isin(val).to_numpy()
    return train[~mask], train[mask]


LOW_DO = 5.0


def tail_weights(target: pd.Series, weight: float) -> np.ndarray:
    """Sample weights: ``weight`` for samples below the 5 mg/L stress threshold, 1 elsewhere."""
    return np.where(target.to_numpy() < LOW_DO, weight, 1.0)


def fit_lgbm(
    train: pd.DataFrame,
    features: list[str],
    params: dict | None = None,
    max_rounds: int = 3000,
    tail_weight: float = 1.0,
) -> lgb.Booster:
    """Fit with early stopping on held-out stations.

    ``tail_weight`` > 1 up-weights low-DO samples in both the training loss and the
    validation score, trading some overall accuracy for better detection of the
    fish-stress regime. Evaluation metrics elsewhere are never weighted.
    """
    fit, val = _carve_validation(train)
    merged = {**DEFAULT_PARAMS, **(params or {})}
    d_fit = lgb.Dataset(fit[features], fit[TARGET], weight=tail_weights(fit[TARGET], tail_weight))
    d_val = lgb.Dataset(
        val[features], val[TARGET], weight=tail_weights(val[TARGET], tail_weight), reference=d_fit
    )
    return lgb.train(
        merged,
        d_fit,
        num_boost_round=max_rounds,
        valid_sets=[d_val],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )


def predict_lgbm(model: lgb.Booster, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(frame[features], num_iteration=model.best_iteration)


def tune_lgbm(
    train: pd.DataFrame,
    features: list[str],
    n_trials: int = 40,
    sample_rows: int = 150_000,
    seed: int = 0,
) -> dict:
    """Optuna TPE search, scored by station-grouped 3-fold RMSE on a training subsample.

    Grouping by station keeps the search honest about the property we care about -
    generalising to stations the model has not seen - rather than rewarding
    memorisation of per-station behaviour.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sub = train.sample(min(sample_rows, len(train)), random_state=seed).reset_index(drop=True)
    folds = list(GroupKFold(3).split(sub, groups=sub["station_id"]))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        }
        merged = {**DEFAULT_PARAMS, **params}
        scores = []
        for tr, va in folds:
            d_tr = lgb.Dataset(sub.loc[tr, features], sub.loc[tr, TARGET])
            d_va = lgb.Dataset(sub.loc[va, features], sub.loc[va, TARGET], reference=d_tr)
            booster = lgb.train(
                merged,
                d_tr,
                num_boost_round=1500,
                valid_sets=[d_va],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            pred = booster.predict(sub.loc[va, features], num_iteration=booster.best_iteration)
            scores.append(float(np.sqrt(np.mean((pred - sub.loc[va, TARGET]) ** 2))))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    log.info("best CV RMSE %.4f after %d trials", study.best_value, n_trials)
    return {"params": study.best_params, "cv_rmse": study.best_value, "n_trials": n_trials}


# --- uncertainty -------------------------------------------------------------------


def conformal_interval(
    model: lgb.Booster,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    level: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-conformal interval from absolute residuals on stations the model did not fit."""
    resid = np.abs(calibration[TARGET].to_numpy() - predict_lgbm(model, calibration, features))
    q = np.quantile(resid, min(1.0, np.ceil((len(resid) + 1) * level) / len(resid)))
    pred = predict_lgbm(model, test, features)
    return pred - q, pred + q


def conformal_quantile(resid: np.ndarray, level: float) -> float:
    return float(np.quantile(resid, min(1.0, np.ceil((len(resid) + 1) * level) / len(resid))))


def binned_conformal_interval(
    calibration_pred: np.ndarray,
    calibration_y: np.ndarray,
    test_pred: np.ndarray,
    edges: tuple[float, ...] = (6.0, 9.0),
    level: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Split-conformal interval with one residual quantile per predicted-DO bin.

    Errors are larger when the predicted DO is low, so a single width over-covers the
    easy cases and under-covers the stressed ones. Binning on the *prediction* (known
    at inference) keeps the finite-sample guarantee within each bin.
    """
    resid = np.abs(calibration_y - calibration_pred)
    cal_bin = np.digitize(calibration_pred, edges)
    test_bin = np.digitize(test_pred, edges)
    widths = np.array(
        [
            conformal_quantile(resid[cal_bin == b], level)
            if (cal_bin == b).sum() > 1 / (1 - level)
            else conformal_quantile(resid, level)
            for b in range(len(edges) + 1)
        ]
    )
    q = widths[test_bin]
    return test_pred - q, test_pred + q


def coverage(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    inside = (y >= lower) & (y <= upper)
    return {"coverage": float(inside.mean()), "mean_width": float((upper - lower).mean())}
