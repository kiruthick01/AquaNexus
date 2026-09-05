"""Synthetic label tests.

These assert the *ecological* properties the labels must have - monotonic decline
under worsening conditions, no penalty for well-oxygenated water, a smooth
interaction term - rather than reproducing whatever numbers the code emits.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.data.synthetic import (
    COLDWATER,
    LOWLAND_WARMWATER,
    HabitatProfile,
    classify,
    falsification_report,
    gaussian_score,
    habitat_suitability,
    interaction_penalty,
    plateau_score,
    sediment_score,
    supersaturation_score,
)


def state(**over):
    row = {"water_temp": 22.0, "dissolved_oxygen": 8.5, "depth": 1.0,
           "velocity": 0.4, "suspended_solids": 12.0}
    row.update(over)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Response curves
# ---------------------------------------------------------------------------


def test_gaussian_peaks_at_optimum_and_is_symmetric():
    assert gaussian_score(10.0, 10.0, 2.0) == pytest.approx(1.0)
    assert gaussian_score(8.0, 10.0, 2.0) == pytest.approx(gaussian_score(12.0, 10.0, 2.0))


def test_plateau_does_not_penalise_excess():
    """The correction: high dissolved oxygen is not harmful."""
    assert plateau_score(8.0, 8.0, 3.0) == pytest.approx(1.0)
    assert plateau_score(12.0, 8.0, 3.0) == pytest.approx(1.0)
    assert plateau_score(17.0, 8.0, 3.0) == pytest.approx(1.0)
    assert plateau_score(4.0, 8.0, 3.0) < 0.5


def test_plateau_matches_gaussian_below_optimum():
    assert plateau_score(5.0, 8.0, 3.0) == pytest.approx(gaussian_score(5.0, 8.0, 3.0))


def test_sediment_score_ramp():
    assert sediment_score(10.0, 25.0, 200.0) == pytest.approx(1.0)
    assert sediment_score(25.0, 25.0, 200.0) == pytest.approx(1.0)
    assert sediment_score(200.0, 25.0, 200.0) == pytest.approx(0.0)
    assert sediment_score(500.0, 25.0, 200.0) == pytest.approx(0.0)  # clipped
    mid = sediment_score(112.5, 25.0, 200.0)
    assert mid == pytest.approx(0.5, abs=0.01)


def test_supersaturation_ignores_mild_excess():
    """106% saturation occurs naturally in the Ayase record and must not be punished."""
    # ~9.1 mg/L is saturation at 20 C; 9.6 is about 106%.
    assert supersaturation_score(9.6, 20.0) == pytest.approx(1.0)


def test_supersaturation_penalises_extreme_excess():
    # 17 mg/L at 20 C is ~187% saturation.
    assert supersaturation_score(17.0, 20.0) < 0.2


def test_interaction_is_smooth_and_monotonic():
    """The spec's if/elif gave worse conditions a smaller penalty."""
    mild = interaction_penalty(0.8, 0.8)
    worse = interaction_penalty(0.5, 0.5)
    worst = interaction_penalty(0.1, 0.1)
    assert mild < worse < worst
    assert interaction_penalty(1.0, 1.0) == pytest.approx(0.0)


def test_interaction_needs_both_stresses():
    """Thermal stress alone carries no synergistic penalty."""
    assert interaction_penalty(0.2, 1.0) == pytest.approx(0.0)
    assert interaction_penalty(1.0, 0.2) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def test_profile_validates_parameters():
    with pytest.raises(ValueError):
        HabitatProfile("bad", 20, 5, 8, 3, 1, 0.5, 0.4, 0.3, 100.0, 50.0)
    with pytest.raises(ValueError):
        HabitatProfile("bad", 20, 0.0, 8, 3, 1, 0.5, 0.4, 0.3, 25.0, 200.0)


def test_unknown_profile_name_raises():
    with pytest.raises(KeyError):
        habitat_suitability(state(), profile="salmon")


def test_profile_lookup_by_name():
    a = habitat_suitability(state(), profile="lowland_warmwater").iloc[0]
    b = habitat_suitability(state(), profile=LOWLAND_WARMWATER).iloc[0]
    assert a == pytest.approx(b)


def test_coldwater_profile_rates_a_warm_river_far_lower():
    """Why the default changed: trout optima make the Ayase uninhabitable."""
    summer = state(water_temp=30.0, dissolved_oxygen=5.0)
    warm = habitat_suitability(summer, LOWLAND_WARMWATER).iloc[0]
    cold = habitat_suitability(summer, COLDWATER).iloc[0]
    assert cold < warm
    assert cold < 0.05


# ---------------------------------------------------------------------------
# Label behaviour
# ---------------------------------------------------------------------------


def test_output_bounded_across_wide_input_sweep():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "water_temp": rng.uniform(0, 40, 400),
        "dissolved_oxygen": rng.uniform(0, 20, 400),
        "depth": rng.uniform(0.05, 6, 400),
        "velocity": rng.uniform(0, 3, 400),
        "suspended_solids": rng.uniform(0, 600, 400),
    })
    hsi = habitat_suitability(frame)
    assert hsi.between(0.0, 1.0).all()
    assert not hsi.isna().any()


def test_good_conditions_score_higher_than_bad():
    good = habitat_suitability(state(water_temp=24.0, dissolved_oxygen=9.0)).iloc[0]
    bad = habitat_suitability(state(water_temp=33.0, dissolved_oxygen=2.5)).iloc[0]
    assert classify([good])[0] == "optimal"
    assert classify([bad])[0] == "unsuitable"


def test_hsi_declines_monotonically_as_oxygen_falls():
    frame = pd.DataFrame({"water_temp": [24.0] * 7,
                          "dissolved_oxygen": [9, 8, 7, 6, 5, 4, 3]})
    hsi = habitat_suitability(frame)
    assert np.all(np.diff(hsi.to_numpy()) <= 1e-12)


def test_hsi_declines_as_temperature_departs_from_optimum():
    frame = pd.DataFrame({"water_temp": [24.0, 28.0, 32.0, 36.0],
                          "dissolved_oxygen": [8.5] * 4})
    hsi = habitat_suitability(frame)
    assert np.all(np.diff(hsi.to_numpy()) < 0)


def test_works_without_depth_and_velocity():
    """Observation frames carry no hydraulics; labels must still generate."""
    frame = pd.DataFrame({"water_temp": [24.0], "dissolved_oxygen": [8.5],
                          "suspended_solids": [12.0]})
    assert 0.0 <= habitat_suitability(frame).iloc[0] <= 1.0


def test_requires_at_least_temperature_or_oxygen():
    with pytest.raises(ValueError, match="temperature or dissolved oxygen"):
        habitat_suitability(pd.DataFrame({"depth": [1.0], "velocity": [0.4]}))


def test_alternative_column_names_resolved():
    frame = pd.DataFrame({"temperature": [24.0], "dissolved_oxygen": [8.5],
                          "suspended_sediment": [12.0]})
    assert habitat_suitability(frame).notna().all()


def test_components_returned_on_request():
    out = habitat_suitability(state(), return_components=True)
    for col in ("temp_score", "do_score", "depth_score", "velocity_score",
                "sediment_score", "supersat_score", "interaction_penalty", "hsi"):
        assert col in out.columns


def test_heavy_sediment_suppresses_an_otherwise_ideal_site():
    ideal = habitat_suitability(state()).iloc[0]
    silted = habitat_suitability(state(suspended_solids=180.0)).iloc[0]
    assert silted < ideal * 0.3


def test_classify_bands():
    bands = classify([0.05, 0.3, 0.5, 0.7, 0.95])
    assert list(bands) == ["unsuitable", "poor", "moderate", "good", "optimal"]


# ---------------------------------------------------------------------------
# Falsification
# ---------------------------------------------------------------------------


def test_falsification_passes_on_a_realistic_gradient():
    frame = pd.DataFrame({
        "water_temp": [8, 12, 18, 22, 26, 29, 31, 32, 20, 15],
        "dissolved_oxygen": [11, 10, 9, 8, 4.5, 3.8, 3.1, 4.4, 8.5, 9.5],
        "suspended_solids": [8, 10, 12, 14, 16, 18, 22, 20, 12, 9],
    })
    report = falsification_report(frame)
    assert report["passed"]
    assert report["hsi_stressed"] < report["hsi_benign"]
    # Correlation is reported but not asserted: temperature confounds it, since
    # for a warmwater guild the coolest samples carry the most oxygen.
    assert report["hsi_low_do"] < report["hsi_high_do"]


def test_falsification_fails_when_the_label_ignores_oxygen():
    """A deliberately broken profile must be caught."""
    blind = HabitatProfile("do_blind", 24.0, 8.0, 0.01, 1e6, 1.0, 0.8, 0.4, 0.35,
                           25.0, 200.0, do_lethal=0.0, do_safe=0.001,
                           interaction_weight=0.0)
    frame = pd.DataFrame({
        "water_temp": [20.0] * 6,
        "dissolved_oxygen": [11, 10, 9, 4.0, 3.5, 3.0],
        "suspended_solids": [10] * 6,
    })
    assert not falsification_report(frame, blind)["passed"]


def test_falsification_reports_insufficient_contrast():
    frame = pd.DataFrame({"water_temp": [20.0, 20.5], "dissolved_oxygen": [8.5, 8.6]})
    report = falsification_report(frame)
    assert not report["passed"]
    assert "reason" in report


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_optimal_secondary_factors_cannot_rescue_lethal_water():
    """Regression: the arithmetic mean scored 33 C / 2.5 mg/L DO at 0.58.

    Ideal depth and velocity contributed 1.0 each and pulled a near-lethal state
    into the "moderate" band. The geometric mean plus the acute-survival term
    must not allow that.
    """
    lethal = state(water_temp=33.0, dissolved_oxygen=2.5, depth=1.0, velocity=0.4)
    assert habitat_suitability(lethal).iloc[0] < 0.25


def test_acute_survival_collapses_hsi_below_the_lethal_threshold():
    from aquanexus.data.synthetic import acute_survival

    assert acute_survival(1.0) == pytest.approx(0.0)
    assert acute_survival(1.5) == pytest.approx(0.0)
    assert acute_survival(2.5) == pytest.approx(0.5)
    assert acute_survival(3.5) == pytest.approx(1.0)
    assert acute_survival(9.0) == pytest.approx(1.0)
    # Salmonids need more oxygen than cyprinids, so the profile sets the bar.
    assert acute_survival(3.0, COLDWATER.do_lethal, COLDWATER.do_safe) == pytest.approx(0.0)

    frame = pd.DataFrame({"water_temp": [24.0], "dissolved_oxygen": [1.4],
                          "depth": [1.0], "velocity": [0.4]})
    assert habitat_suitability(frame).iloc[0] == pytest.approx(0.0)


def test_geometric_mean_is_dragged_down_by_one_poor_factor():
    ideal = habitat_suitability(state()).iloc[0]
    one_bad = habitat_suitability(state(velocity=2.5)).iloc[0]
    assert one_bad < ideal * 0.6
