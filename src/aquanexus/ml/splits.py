"""Train/validation/test splits.

ML_STRATEGY.md §5 specifies three splits, all built on a 4-year hourly series
that does not exist - see ``docs/ML_METHODOLOGY.md``. They are replaced here:

======================  =========================================================
§5.1 Temporal           **Not applicable.** The dataset is a designed experiment
                        over flow space, not a chronology. There is no "future"
                        to hold out. A date-based split is still offered as
                        :func:`temporal_split`, because the *observations* do
                        carry dates and holding out a season is a meaningful
                        test - but it is a different claim from the one §5.1
                        makes.
§5.2 Spatial            **Kept.** Hold out cross-sections; the model must work
                        on channel geometry it has not seen.
§5.3 Event-based        **Reframed** as :func:`flow_split`: train on the middle
                        of the discharge range, test on the extremes. Same
                        intent - does the model behave at conditions it was not
                        trained on - without pretending to have flood events.
======================  =========================================================

Every split here is **grouped**. All 53 rows from one observation share the same
water chemistry, so a random split puts near-duplicates on both sides and returns
a score that is mostly memorisation. :func:`random_split` exists to demonstrate
exactly that inflation, not for use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("ml.splits")


@dataclass
class Split:
    """Index masks for one train/validation/test division."""

    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    note: str = ""

    def sizes(self) -> dict[str, int]:
        return {"train": int(self.train.sum()),
                "validation": int(self.validation.sum()),
                "test": int(self.test.sum())}

    def apply(self, frame: pd.DataFrame, features: list[str], target: str = "hsi"):
        """Return ``(X_train, y_train), (X_val, y_val), (X_test, y_test)``."""
        out = []
        for mask in (self.train, self.validation, self.test):
            part = frame.loc[mask]
            out.append((part[features], part[target]))
        return tuple(out)

    def __str__(self) -> str:
        sizes = self.sizes()
        total = sum(sizes.values()) or 1
        parts = ", ".join(f"{k} {v} ({v/total:.0%})" for k, v in sizes.items())
        return f"{self.name}: {parts}" + (f" - {self.note}" if self.note else "")


def _check_disjoint(frame: pd.DataFrame, split: Split, key: str) -> None:
    """Fail loudly if a grouping key appears on both sides of a split."""
    train_keys = set(frame.loc[split.train, key].unique())
    test_keys = set(frame.loc[split.test, key].unique())
    overlap = train_keys & test_keys
    if overlap:
        raise ValueError(
            f"{split.name}: {len(overlap)} value(s) of '{key}' appear in both "
            f"train and test; the split is leaking"
        )


def grouped_split(
    frame: pd.DataFrame,
    group_column: str = "group",
    test_size: float = 0.2,
    validation_size: float = 0.15,
    seed: int = 42,
) -> Split:
    """Hold out whole observations.

    The default split: rows sharing an observation share its chemistry, so they
    must not straddle the boundary.
    """
    groups = np.sort(frame[group_column].unique())
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(groups)

    n_test = max(1, int(round(len(groups) * test_size)))
    n_val = max(1, int(round(len(groups) * validation_size)))
    test_groups = set(shuffled[:n_test])
    val_groups = set(shuffled[n_test:n_test + n_val])

    values = frame[group_column]
    split = Split(
        name="grouped",
        train=~values.isin(test_groups | val_groups),
        validation=values.isin(val_groups),
        test=values.isin(test_groups),
        note=f"{len(groups)} observation groups held out whole",
    )
    _check_disjoint(frame, split, group_column)
    log.info("%s", split)
    return split


def spatial_split(
    frame: pd.DataFrame,
    section_column: str = "river_station",
    test_fraction: float = 0.3,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> Split:
    """Hold out whole cross-sections (ML_STRATEGY §5.2).

    Sections are held out in contiguous blocks rather than at random. Scattering
    them leaves every test section between two training neighbours, which is a
    much easier problem than a genuinely unseen stretch of river.
    """
    sections = np.sort(frame[section_column].unique())
    n_test = max(1, int(round(len(sections) * test_fraction)))
    n_val = max(1, int(round(len(sections) * validation_fraction)))

    rng = np.random.default_rng(seed)
    start = int(rng.integers(0, max(1, len(sections) - n_test - n_val)))
    test_sections = set(sections[start:start + n_test])
    val_sections = set(sections[start + n_test:start + n_test + n_val])

    values = frame[section_column]
    split = Split(
        name="spatial",
        train=~values.isin(test_sections | val_sections),
        validation=values.isin(val_sections),
        test=values.isin(test_sections),
        note=f"contiguous block of {len(test_sections)} of {len(sections)} sections held out",
    )
    _check_disjoint(frame, split, section_column)
    log.info("%s", split)
    return split


def flow_split(
    frame: pd.DataFrame,
    discharge_column: str = "discharge",
    low_quantile: float = 0.15,
    high_quantile: float = 0.85,
) -> Split:
    """Train on mid-range discharge, test on both extremes (reframed §5.3).

    Extrapolation is the honest test here: the model sees ordinary flows and is
    asked about droughts and floods. Expect it to do worse than on a grouped
    split - that gap is the finding, not a failure.
    """
    discharge = frame[discharge_column]
    low, high = discharge.quantile(low_quantile), discharge.quantile(high_quantile)

    test = (discharge <= low) | (discharge >= high)
    middle = ~test
    # Carve validation from the interior edges so it is harder than the core.
    interior = discharge[middle]
    val_low, val_high = interior.quantile(0.1), interior.quantile(0.9)
    validation = middle & ((discharge <= val_low) | (discharge >= val_high))

    split = Split(
        name="flow",
        train=middle & ~validation,
        validation=validation,
        test=test,
        note=f"test on discharge <= {low:.2f} or >= {high:.2f} m3/s",
    )
    log.info("%s", split)
    return split


def temporal_split(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
    test_fraction: float = 0.2,
    validation_fraction: float = 0.15,
) -> Split:
    """Hold out the most recent observations.

    Offered because the observations do carry dates, but note this is **not**
    ML_STRATEGY §5.1: that split assumed a continuous hourly series and claimed
    to test forecasting. This tests whether the model transfers to later
    sampling dates, which is a weaker and different claim.
    """
    if timestamp_column not in frame.columns:
        raise ValueError(f"no '{timestamp_column}' column")

    stamps = pd.to_datetime(frame[timestamp_column])
    order = np.sort(stamps.unique())
    n_test = max(1, int(round(len(order) * test_fraction)))
    n_val = max(1, int(round(len(order) * validation_fraction)))

    test_from = order[-n_test]
    val_from = order[-(n_test + n_val)]

    split = Split(
        name="temporal",
        train=stamps < val_from,
        validation=(stamps >= val_from) & (stamps < test_from),
        test=stamps >= test_from,
        note=f"test from {pd.Timestamp(test_from).date()} onward",
    )
    log.info("%s", split)
    return split


def random_split(frame: pd.DataFrame, test_size: float = 0.2,
                 validation_size: float = 0.15, seed: int = 42) -> Split:
    """Ungrouped random split - **for demonstrating leakage, not for use**.

    Reported alongside the grouped split, the gap between them measures how much
    of an ungrouped score is memorisation of near-duplicate rows.
    """
    rng = np.random.default_rng(seed)
    draw = rng.random(len(frame))
    return Split(
        name="random (leaky)",
        train=pd.Series(draw >= test_size + validation_size, index=frame.index),
        validation=pd.Series((draw >= test_size) & (draw < test_size + validation_size),
                             index=frame.index),
        test=pd.Series(draw < test_size, index=frame.index),
        note="ungrouped; inflated by design",
    )
