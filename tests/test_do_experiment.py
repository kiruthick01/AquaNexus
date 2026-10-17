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


def test_tail_weights_upweight_only_samples_below_the_stress_threshold():
    target = pd.Series([2.0, 4.99, 5.0, 9.0])
    assert ex.tail_weights(target, 10.0).tolist() == [10.0, 10.0, 1.0, 1.0]


def test_binned_conformal_is_wider_where_calibration_errors_are_larger():
    rng = np.random.default_rng(0)
    pred = np.concatenate([np.full(500, 4.0), np.full(500, 10.0)])
    noise = np.concatenate([rng.normal(0, 2.0, 500), rng.normal(0, 0.3, 500)])
    lo, hi = ex.binned_conformal_interval(pred, pred + noise, np.array([4.0, 10.0]))
    assert (hi - lo)[0] > 3 * (hi - lo)[1]


def test_binned_conformal_covers_near_nominal_on_fresh_data():
    rng = np.random.default_rng(1)

    def draw(n):
        pred = rng.uniform(2, 12, n)
        return pred, pred + rng.normal(0, 0.2 + 0.2 * (12 - pred))

    cal_pred, cal_y = draw(6000)
    pred, y = draw(6000)
    lo, hi = ex.binned_conformal_interval(cal_pred, cal_y, pred, level=0.9)
    assert ex.coverage(y, lo, hi)["coverage"] == pytest.approx(0.9, abs=0.03)


def test_do_bundle_round_trips_and_returns_ordered_intervals(tmp_path):
    from aquanexus.ml.do_bundle import DOBundle

    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    frame["dissolved_oxygen"] = 8 + frame["a"]
    booster = ex.lgb.train(
        {**ex.DEFAULT_PARAMS, "num_leaves": 7},
        ex.lgb.Dataset(frame[["a", "b"]], frame["dissolved_oxygen"]),
        num_boost_round=20,
    )
    bundle = DOBundle(booster, ["a", "b"], [6.0, 9.0], [1.5, 1.0, 0.8], 0.9, {"name": "t"})
    bundle.save(tmp_path / "m")
    loaded = DOBundle.load(tmp_path / "m")
    out = loaded.predict(frame)
    pd.testing.assert_frame_equal(out, bundle.predict(frame), check_exact=False, atol=1e-6)
    assert (out["lower"] < out["prediction"]).all() and (out["prediction"] < out["upper"]).all()
    with pytest.raises(KeyError):
        loaded.predict(frame[["a"]])


def _toy_frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    y = 4 + 8 * x + rng.normal(0, 0.5 + 1.5 * (1 - x), n)  # noisier when x is small
    return pd.DataFrame({"x": x, "dissolved_oxygen": y, "station_id": rng.integers(0, 60, n)})


def test_cqr_reaches_nominal_coverage_and_is_wider_where_noise_is_larger():
    frame = _toy_frame()
    train, cal, test = frame.iloc[:1500], frame.iloc[1500:2200], frame.iloc[2200:]
    lo = ex.fit_quantile(train, ["x"], 0.05)
    hi = ex.fit_quantile(train, ["x"], 0.95)
    lower, upper = ex.cqr_interval(lo, hi, cal, test, ["x"], level=0.9)
    cover = ex.coverage(test["dissolved_oxygen"].to_numpy(), lower, upper)
    assert cover["coverage"] == pytest.approx(0.9, abs=0.05)
    width = upper - lower
    assert width[test["x"].to_numpy() < 0.2].mean() > width[test["x"].to_numpy() > 0.8].mean()


def test_ranking_metrics_separate_a_good_score_from_a_random_one():
    rng = np.random.default_rng(0)
    y = np.where(rng.uniform(size=2000) < 0.2, 3.0, 8.0)
    good = ex.ranking_metrics(y, -y + rng.normal(0, 1.0, len(y)))
    random = ex.ranking_metrics(y, rng.normal(size=len(y)))
    assert good["auc"] > 0.9 and abs(random["auc"] - 0.5) < 0.08
    assert good["avg_precision"] > random["avg_precision"] + 0.3


def test_low_do_classifier_outputs_probabilities():
    frame = _toy_frame()
    model = ex.fit_low_do_classifier(frame, ["x"])
    prob = model.predict(frame[["x"]], num_iteration=model.best_iteration)
    assert ((prob >= 0) & (prob <= 1)).all()
    assert prob[frame["x"] < 0.1].mean() > prob[frame["x"] > 0.9].mean()


def test_isotonic_map_repairs_an_overconfident_probability():
    rng = np.random.default_rng(0)
    truth = rng.uniform(size=4000) < 0.2
    y = np.where(truth, 3.0, 8.0)
    overconfident = np.clip(0.2 + (truth - 0.2) * 0.9 * rng.uniform(0.3, 1.0, 4000), 0, 1)
    shifted = np.clip(overconfident * 1.0 + 0.3, 0, 1)  # systematically too high
    mapping = ex.isotonic_map(shifted, y)
    fixed = np.interp(shifted, mapping["x"], mapping["y"])
    assert ex.calibration_report(fixed, y)["ece"] < ex.calibration_report(shifted, y)["ece"]
    assert (np.diff(mapping["y"]) >= 0).all()
