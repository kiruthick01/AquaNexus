"""Tests for the DO experiment: splits must not leak, baselines must use training data only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aquanexus.ml import do_experiment as ex


@pytest.fixture()
def frame():
    rng = np.random.default_rng(1)
    rows = []
    for station in range(1, 41):
        for fy in range(2013, 2023):
            for month in (1, 5, 9):
                rows.append(
                    {
                        "station_id": station,
                        "pref_code": station % 5 + 1,
                        "fiscal_year": fy,
                        "month": month,
                        "dissolved_oxygen": 9 + rng.normal() - 0.2 * (month == 9),
                    }
                )
    out = pd.DataFrame(rows)
    out["prev_do"] = out.groupby("station_id")["dissolved_oxygen"].shift(1)
    return out


def test_splits_are_disjoint_and_strict_is_both_unseen_stations_and_later_years(frame):
    splits = {s.name: s for s in ex.make_splits(frame)}
    for split in splits.values():
        assert not (split.train & split.test).any()

    strict = splits["strict"]
    train, test = frame[strict.train], frame[strict.test]
    assert set(train["station_id"]).isdisjoint(test["station_id"])
    assert train["fiscal_year"].max() <= ex.TRAIN_LAST_FY
    assert test["fiscal_year"].min() >= ex.TEST_FIRST_FY


def test_temporal_split_purges_the_gap_year(frame):
    temporal = {s.name: s for s in ex.make_splits(frame)}["temporal"]
    used = set(frame[temporal.train | temporal.test]["fiscal_year"])
    assert 2020 not in used


def test_station_split_holds_out_about_a_fifth(frame):
    station = {s.name: s for s in ex.make_splits(frame)}["station"]
    held = frame[station.test]["station_id"].nunique()
    assert held == 8


def test_prefecture_folds_cover_every_row_once_and_never_share_a_prefecture(frame):
    folds = ex.prefecture_folds(frame, n_splits=5)
    assert sum(f.test.sum() for f in folds) == len(frame)
    for fold in folds:
        assert set(frame[fold.train]["pref_code"]).isdisjoint(frame[fold.test]["pref_code"])


def test_climatology_uses_only_training_rows(frame):
    train = frame[frame["fiscal_year"] <= 2019]
    test = frame[frame["fiscal_year"] >= 2021].copy()
    test["dissolved_oxygen"] = 999.0  # a leak would drag predictions toward this
    assert (ex.climatology(train, test) < 20).all()


def test_climatology_falls_back_for_unseen_stations(frame):
    train = frame[frame["station_id"] <= 30]
    test = frame[frame["station_id"] > 30]
    pred = ex.climatology(train, test)
    assert not np.isnan(pred).any()


def test_metrics_on_a_perfect_and_a_biased_prediction():
    y = np.array([3.0, 4.0, 8.0, 9.0])
    perfect = ex.metrics(y, y)
    assert perfect["r2"] == pytest.approx(1.0)
    assert perfect["recall_low"] == 1.0
    biased = ex.metrics(y, y + 2.0)
    assert biased["bias"] == pytest.approx(2.0)
    assert biased["mae_low"] == pytest.approx(2.0)


def test_metrics_ignore_missing_predictions():
    out = ex.metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]))
    assert out["n"] == 2
