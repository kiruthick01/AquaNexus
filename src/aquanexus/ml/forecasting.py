"""Temporal feature and split infrastructure for forecasting - **not yet applicable**.

``docs/ML_ROADMAP.md`` Phase 1 and ``docs/DATA_LIMITATIONS.md`` establish why: the
138 real dissolved-oxygen observations are irregular grab samples (roughly monthly,
gaps of weeks to months) rather than a regular series. A lag feature built from
whatever row happens to precede another is not "yesterday's value" - it might be six
weeks old - and reporting a forecast built on that would misrepresent the model's
actual horizon.

Everything here is built and tested against synthetic regular series only
(``tests/test_forecasting.py``). It exists so that if a denser, regularly-sampled
record becomes available, forecasting can be built on tested infrastructure rather
than from scratch - see ``scripts/phase1_forecasting_feasibility.py`` for the
concrete evidence, run against the real dataset, that no usable forecasting sample
currently exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("ml.forecasting")


def _as_days(offset: str | float) -> float:
    """Normalise a lag/window given as days or a pandas offset string to days."""
    if isinstance(offset, str):
        return pd.Timedelta(offset) / pd.Timedelta(days=1)
    return float(offset)


def add_lag_features(
    frame: pd.DataFrame,
    value_column: str,
    timestamp_column: str = "timestamp",
    group_column: str = "station",
    lags: tuple[float, ...] = (1, 3, 7),
    tolerance_fraction: float = 0.2,
    min_tolerance_days: float = 1.0,
) -> pd.DataFrame:
    """Add ``{value_column}_lag{N}`` columns, each NaN unless a real observation
    exists close enough in time to call it "N days ago".

    A relative tolerance is used, not a fixed number of days, because the lags
    themselves span an order of magnitude (1 to 7+ days): a tolerance tight enough
    to be honest at lag 1 would reject nearly everything at lag 7, and one loose
    enough for lag 7 would let a lag-1 feature be built from a four-day-old sample.
    ``tolerance_fraction=0.2`` means a lag is accepted only if the true gap is
    within +/-20% of the nominal lag, floored at ``min_tolerance_days`` so a
    lag-1 request is not held to a sub-day tolerance no grab-sample record can
    meet. Both are explicit parameters, not implicit defaults, because the right
    tolerance is a property of the sampling design, not a universal constant.

    Rows are sorted chronologically within each group first; the function never
    uses a future row to fill a past one.
    """
    if timestamp_column not in frame.columns:
        raise ValueError(f"no '{timestamp_column}' column")
    if value_column not in frame.columns:
        raise ValueError(f"no '{value_column}' column")

    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column])
    out = out.sort_values([group_column, timestamp_column]).reset_index(drop=True)

    for lag in lags:
        lag_days = _as_days(lag)
        tolerance = max(min_tolerance_days, tolerance_fraction * lag_days)
        col = f"{value_column}_lag{lag_days:g}d"
        gap_col = f"{value_column}_lag{lag_days:g}d_gap_days"
        values = np.full(len(out), np.nan)
        gaps = np.full(len(out), np.nan)

        for _, block in out.groupby(group_column, sort=False):
            idx = block.index.to_numpy()
            stamps = block[timestamp_column].to_numpy()
            series = block[value_column].to_numpy(dtype=float)
            for i in range(len(idx)):
                # nearest earlier-or-equal observation to the target lag time
                candidates = np.where(stamps[:i] <= stamps[i])[0]
                if len(candidates) == 0:
                    continue
                prior_stamps = stamps[candidates]
                deltas_days = (stamps[i] - prior_stamps) / np.timedelta64(1, "D")
                closeness = np.abs(deltas_days - lag_days)
                best = candidates[np.argmin(closeness)]
                best_gap = float((stamps[i] - stamps[best]) / np.timedelta64(1, "D"))
                if abs(best_gap - lag_days) <= tolerance:
                    values[idx[i]] = series[best]
                    gaps[idx[i]] = best_gap

        out[col] = values
        out[gap_col] = gaps
        n_usable = int(np.isfinite(values).sum())
        log.info("lag %sd: %d/%d row(s) within tolerance +/-%.1fd", lag, n_usable, len(out),
                 tolerance)

    return out


def add_lead_targets(
    frame: pd.DataFrame,
    value_column: str,
    timestamp_column: str = "timestamp",
    group_column: str = "station",
    horizons: tuple[float, ...] = (1, 3, 7),
    tolerance_fraction: float = 0.2,
    min_tolerance_days: float = 1.0,
) -> pd.DataFrame:
    """Add ``{value_column}_lead{H}d`` columns - the forecasting targets
    DO(t+H) - each NaN unless a real future observation exists close enough to
    ``H`` days ahead. Mirrors :func:`add_lag_features` but looks forward instead
    of backward, with the same relative-tolerance reasoning: the target for a
    "next-day" forecast must actually be next-day, not whatever happens to be
    the next grab sample weeks later.
    """
    if timestamp_column not in frame.columns:
        raise ValueError(f"no '{timestamp_column}' column")
    if value_column not in frame.columns:
        raise ValueError(f"no '{value_column}' column")

    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column])
    out = out.sort_values([group_column, timestamp_column]).reset_index(drop=True)

    for horizon in horizons:
        horizon_days = _as_days(horizon)
        tolerance = max(min_tolerance_days, tolerance_fraction * horizon_days)
        col = f"{value_column}_lead{horizon_days:g}d"
        gap_col = f"{value_column}_lead{horizon_days:g}d_gap_days"
        values = np.full(len(out), np.nan)
        gaps = np.full(len(out), np.nan)

        for _, block in out.groupby(group_column, sort=False):
            idx = block.index.to_numpy()
            stamps = block[timestamp_column].to_numpy()
            series = block[value_column].to_numpy(dtype=float)
            for i in range(len(idx)):
                candidates = np.where(stamps[i + 1:] >= stamps[i])[0] + (i + 1)
                if len(candidates) == 0:
                    continue
                future_stamps = stamps[candidates]
                deltas_days = (future_stamps - stamps[i]) / np.timedelta64(1, "D")
                closeness = np.abs(deltas_days - horizon_days)
                best_pos = int(np.argmin(closeness))
                best = candidates[best_pos]
                best_gap = float((stamps[best] - stamps[i]) / np.timedelta64(1, "D"))
                if abs(best_gap - horizon_days) <= tolerance:
                    values[idx[i]] = series[best]
                    gaps[idx[i]] = best_gap

        out[col] = values
        out[gap_col] = gaps
        n_usable = int(np.isfinite(values).sum())
        log.info("lead %sd: %d/%d row(s) within tolerance +/-%.1fd", horizon, n_usable,
                 len(out), tolerance)

    return out


def add_rolling_features(
    frame: pd.DataFrame,
    value_column: str,
    timestamp_column: str = "timestamp",
    group_column: str = "station",
    window: str = "30D",
    min_periods: int = 2,
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Add time-based rolling statistics of ``value_column``, per group.

    A calendar-time window (``window="30D"``) is used rather than a row-count
    window: with irregular sampling, "the last 3 rows" can span anywhere from a
    week to most of a year, so a fixed row count does not mean a fixed amount of
    history. Each rolling value only ever looks backward from its own timestamp
    (pandas' default, left-closed-on-the-past rolling), so no future observation
    can leak into it.
    """
    if timestamp_column not in frame.columns:
        raise ValueError(f"no '{timestamp_column}' column")
    if value_column not in frame.columns:
        raise ValueError(f"no '{value_column}' column")

    out = frame.copy()
    out[timestamp_column] = pd.to_datetime(out[timestamp_column])
    out = out.sort_values([group_column, timestamp_column]).reset_index(drop=True)

    results = {stat: np.full(len(out), np.nan) for stat in stats}
    for _, block in out.groupby(group_column, sort=False):
        indexed = block.set_index(timestamp_column)[value_column]
        rolled = indexed.rolling(window, min_periods=min_periods)
        for stat in stats:
            results[stat][block.index.to_numpy()] = getattr(rolled, stat)().to_numpy()

    for stat in stats:
        out[f"{value_column}_roll{window}_{stat}"] = results[stat]

    return out


@dataclass
class WalkForwardFold:
    """One chronological fold: strictly-past train, strictly-future test."""

    fold: int
    train: np.ndarray
    test: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp


def walk_forward_splits(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
    group_column: str = "station",
    n_folds: int = 5,
) -> list[WalkForwardFold]:
    """Expanding-window chronological folds, computed independently per group.

    Fold ``k`` trains on everything before a cut point and tests on the next
    slice of that same group's timeline; the cut points advance forward, and
    rows are never shuffled. A group with too few distinct timestamps to form
    ``n_folds`` non-empty folds contributes fewer folds rather than an empty or
    duplicated one - silently padding a fold would misrepresent how much
    validation evidence actually exists.
    """
    if timestamp_column not in frame.columns:
        raise ValueError(f"no '{timestamp_column}' column")

    stamps = pd.to_datetime(frame[timestamp_column])
    folds: list[WalkForwardFold] = []

    for group, block in frame.groupby(group_column, sort=False):
        block_stamps = stamps.loc[block.index]
        unique_times = np.sort(block_stamps.unique())
        if len(unique_times) < n_folds + 1:
            log.info("group %s: %d unique timestamp(s), too few for %d folds - skipped",
                     group, len(unique_times), n_folds)
            continue

        boundaries = np.array_split(np.arange(1, len(unique_times)), n_folds)
        for k, test_positions in enumerate(boundaries):
            if len(test_positions) == 0:
                continue
            cut = unique_times[test_positions[0]]
            test_times = unique_times[test_positions]
            train_idx = block_stamps[block_stamps < cut].index
            test_idx = block_stamps[block_stamps.isin(test_times)].index

            full_train = pd.Series(False, index=frame.index)
            full_test = pd.Series(False, index=frame.index)
            full_train.loc[train_idx] = True
            full_test.loc[test_idx] = True
            if full_train.sum() == 0 or full_test.sum() == 0:
                continue
            folds.append(WalkForwardFold(
                fold=k,
                train=full_train.to_numpy(),
                test=full_test.to_numpy(),
                train_end=pd.Timestamp(cut),
                test_start=pd.Timestamp(test_times.min()),
            ))

    return folds


def assert_no_temporal_leakage(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    timestamp_column: str = "timestamp",
    group_column: str = "station",
) -> None:
    """Raise ``ValueError`` if any test row's timestamp is <= a train row's
    timestamp within the same group.

    This is the check every walk-forward split in this module must pass; it is
    exposed separately so a hand-built or externally-supplied split can be
    verified the same way.
    """
    stamps = pd.to_datetime(frame[timestamp_column])
    groups = frame[group_column]

    train_frame = pd.DataFrame({"t": stamps[train_mask], "g": groups[train_mask]})
    test_frame = pd.DataFrame({"t": stamps[test_mask], "g": groups[test_mask]})

    max_train_by_group = train_frame.groupby("g")["t"].max()
    violations = []
    for group, test_block in test_frame.groupby("g"):
        if group not in max_train_by_group.index:
            continue
        latest_train = max_train_by_group.loc[group]
        leaked = test_block[test_block["t"] <= latest_train]
        if not leaked.empty:
            violations.append((group, len(leaked), latest_train))

    if violations:
        detail = "; ".join(
            f"group {g}: {n} test row(s) <= last train timestamp {t}" for g, n, t in violations
        )
        raise ValueError(f"temporal leakage detected: {detail}")
