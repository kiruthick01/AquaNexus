"""Feature engineering tests.

The physical relationships are checked against published reference values rather
than against whatever the implementation happens to return, so a regression in
the constants is caught.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.data.preprocessor import (
    AMPLE_DO,
    CRITICAL_DO,
    OPTIMAL_TEMP_C,
    add_derived_features,
    bed_shear_stress,
    combined_stress,
    do_saturation,
    froude_number,
    kinematic_viscosity,
    observed_ranges,
    oxygen_stress,
    reynolds_number,
    thermal_stress,
)

# APHA Standard Methods 4500-O saturation table, fresh water at 1 atm (mg/L).
APHA_TABLE = {0: 14.62, 5: 12.77, 10: 11.29, 15: 10.08, 20: 9.09, 25: 8.26,
              30: 7.56, 35: 6.95}


# ---------------------------------------------------------------------------
# Dissolved oxygen saturation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("temp,expected", sorted(APHA_TABLE.items()))
def test_do_saturation_matches_published_table(temp, expected):
    assert do_saturation(temp) == pytest.approx(expected, abs=0.03)


def test_do_saturation_decreases_with_temperature():
    temps = np.arange(0, 40, 2.0)
    assert np.all(np.diff(do_saturation(temps)) < 0)


def test_do_saturation_outside_valid_range_is_nan():
    assert np.isnan(do_saturation(-5.0))
    assert np.isnan(do_saturation(45.0))


def test_elevation_reduces_saturation():
    sea = do_saturation(20.0, elevation_m=0.0)
    high = do_saturation(20.0, elevation_m=1000.0)
    assert high < sea
    assert high == pytest.approx(sea * np.exp(-1000 / 8434), rel=1e-9)


def test_saturation_is_vectorised():
    out = do_saturation(np.array([10.0, 20.0, 30.0]))
    assert out.shape == (3,)
    assert out[0] > out[1] > out[2]


# ---------------------------------------------------------------------------
# Hydraulics
# ---------------------------------------------------------------------------


def test_kinematic_viscosity_reference_values():
    # ~1.79e-6 at 0 C falling to ~1.00e-6 at 20 C.
    assert kinematic_viscosity(0.0) == pytest.approx(1.79e-6, rel=0.02)
    assert kinematic_viscosity(20.0) == pytest.approx(1.00e-6, rel=0.05)
    assert kinematic_viscosity(30.0) < kinematic_viscosity(10.0)


def test_froude_critical_point():
    # At v = sqrt(g*d) the flow is exactly critical.
    depth = 2.0
    assert froude_number(np.sqrt(9.80665 * depth), depth) == pytest.approx(1.0)


def test_froude_subcritical_for_typical_lowland_flow():
    # Ayase-like: slow and deep.
    assert froude_number(0.4, 2.5) < 1.0


def test_froude_zero_depth_is_nan():
    assert np.isnan(froude_number(1.0, 0.0))


def test_reynolds_turbulent_for_river_flow():
    # Anything above ~2000 is turbulent; a real river is far above.
    assert reynolds_number(0.5, 2.0, 15.0) > 1e5


def test_reynolds_rises_as_water_warms():
    """Viscosity falls with temperature, so Re rises for the same flow."""
    cold = reynolds_number(0.5, 2.0, 5.0)
    warm = reynolds_number(0.5, 2.0, 30.0)
    assert warm > cold


def test_shear_stress_formula():
    # tau = rho*g*d*S = 1000 * 9.80665 * 2.0 * 1e-4
    assert bed_shear_stress(2.0, 1e-4) == pytest.approx(1.96133, rel=1e-4)


def test_shear_stress_scales_with_depth_and_slope():
    assert bed_shear_stress(4.0, 1e-4) == pytest.approx(2 * bed_shear_stress(2.0, 1e-4))
    assert bed_shear_stress(2.0, 2e-4) == pytest.approx(2 * bed_shear_stress(2.0, 1e-4))


# ---------------------------------------------------------------------------
# Stress indices
# ---------------------------------------------------------------------------


def test_thermal_stress_is_zero_at_optimum():
    assert thermal_stress(OPTIMAL_TEMP_C) == pytest.approx(0.0, abs=1e-12)


def test_thermal_stress_penalises_cold_as_well_as_heat():
    """The correction to the spec: its sigmoid never penalised cold water."""
    cold = thermal_stress(OPTIMAL_TEMP_C - 12.0)
    warm = thermal_stress(OPTIMAL_TEMP_C + 12.0)
    assert cold > 0.5
    assert cold == pytest.approx(warm)  # symmetric about the optimum


def test_thermal_stress_bounded():
    values = thermal_stress(np.linspace(-5, 45, 60))
    assert np.all((values >= 0.0) & (values <= 1.0))


def test_oxygen_stress_endpoints():
    assert oxygen_stress(AMPLE_DO) == pytest.approx(0.0)
    assert oxygen_stress(CRITICAL_DO) == pytest.approx(1.0)
    assert oxygen_stress(0.0) == pytest.approx(1.0)  # clipped, not extrapolated


def test_oxygen_stress_bounded_above_ample():
    """The spec's exp() form grew without limit above the optimum."""
    values = oxygen_stress(np.array([9.0, 12.0, 20.0]))
    assert np.all(values == 0.0)


def test_oxygen_stress_is_monotonic_and_convex():
    do = np.linspace(CRITICAL_DO, AMPLE_DO, 25)
    s = oxygen_stress(do)
    assert np.all(np.diff(s) < 0)
    # Squared response: the midpoint sits below the linear interpolation.
    assert oxygen_stress((CRITICAL_DO + AMPLE_DO) / 2) < 0.5


def test_combined_stress_bounds():
    assert combined_stress(0.0, 0.0) == pytest.approx(0.0)
    assert combined_stress(1.0, 1.0) == pytest.approx(1.0)
    assert 0.0 < combined_stress(0.5, 0.5) < 1.0


# ---------------------------------------------------------------------------
# Frame-level
# ---------------------------------------------------------------------------


@pytest.fixture
def observations():
    return pd.DataFrame(
        {
            "water_temp": [21.0, 26.1, 31.0, 8.0],
            "dissolved_oxygen": [8.3, 3.6, 4.2, 11.5],
            "discharge": [22.58, 50.05, 63.74, 5.0],
            "suspended_solids": [13.0, 16.0, 22.0, 8.0],
            "depth": [1.8, 2.4, 2.6, 1.1],
            "velocity": [0.35, 0.55, 0.62, 0.2],
        }
    )


def test_derived_columns_present(observations):
    out = add_derived_features(observations)
    for col in ("do_saturation", "do_deficit", "do_saturation_pct", "froude_number",
                "reynolds_number", "shear_stress", "thermal_stress_index",
                "oxygen_stress_index", "combined_stress_index", "temp_x_do",
                "flow_x_sediment", "discharge_x_temp"):
        assert col in out.columns


def test_do_deficit_positive_when_undersaturated(observations):
    out = add_derived_features(observations)
    # The 26.1 C / 3.6 mg/L sample is heavily undersaturated.
    assert out["do_deficit"].iloc[1] > 4.0
    assert out["do_saturation_pct"].iloc[1] < 50.0


def test_hot_low_oxygen_sample_is_most_stressed(observations):
    out = add_derived_features(observations)
    assert out["combined_stress_index"].idxmax() in (1, 2)
    assert out["combined_stress_index"].iloc[1] > out["combined_stress_index"].iloc[0]


def test_missing_inputs_are_skipped_not_fatal():
    frame = pd.DataFrame({"water_temp": [20.0, 25.0]})
    out = add_derived_features(frame)
    assert "thermal_stress_index" in out.columns
    assert "froude_number" not in out.columns
    assert "do_deficit" not in out.columns


def test_original_frame_not_mutated(observations):
    before = list(observations.columns)
    add_derived_features(observations)
    assert list(observations.columns) == before


def test_slope_can_be_a_column(observations):
    frame = observations.assign(slope=[1e-4, 2e-4, 1e-4, 3e-4])
    out = add_derived_features(frame, slope="slope")
    assert out["shear_stress"].iloc[1] == pytest.approx(
        bed_shear_stress(2.4, 2e-4), rel=1e-9
    )


def test_attrs_preserved(observations):
    observations.attrs["units"] = {"dissolved_oxygen": "mg/L"}
    out = add_derived_features(observations)
    assert out.attrs["units"]["dissolved_oxygen"] == "mg/L"


# --- ranges -----------------------------------------------------------------


def test_observed_ranges_shape_and_columns(observations):
    stats = observed_ranges(observations)
    assert set(stats.columns) == {"count", "mean", "std", "min", "q0.01", "q0.99", "max"}
    assert "water_temp" in stats.index


def test_observed_ranges_quantiles_resist_outliers():
    """The upper quantile must not be dragged out to the tail the way max is.

    With one wild value in 100 the interpolated 99th percentile still moves a
    little, so the claim under test is the order-of-magnitude gap from max, not
    that the quantile is entirely unaffected.
    """
    frame = pd.DataFrame({"discharge": list(np.linspace(1, 50, 99)) + [100000.0]})
    stats = observed_ranges(frame)
    assert stats.loc["discharge", "max"] == 100000.0
    assert stats.loc["discharge", "q0.99"] < stats.loc["discharge", "max"] / 50


def test_observed_ranges_quantiles_stable_with_more_data():
    """With a realistic sample size one outlier stops mattering entirely."""
    frame = pd.DataFrame({"discharge": list(np.linspace(1, 50, 999)) + [100000.0]})
    stats = observed_ranges(frame)
    assert stats.loc["discharge", "q0.99"] < 60.0


def test_observed_ranges_subset_and_missing(observations, caplog):
    stats = observed_ranges(observations, columns=["water_temp", "nope"])
    assert list(stats.index) == ["water_temp"]


def test_observed_ranges_ignores_non_numeric():
    frame = pd.DataFrame({"station": ["a", "b"], "water_temp": [10.0, 20.0]})
    assert list(observed_ranges(frame).index) == ["water_temp"]
