"""Feature engineering for the habitat suitability model.

Turns raw hydraulic and water quality state into the model's feature set:
physically-derived quantities (dissolved oxygen saturation and deficit, Froude
and Reynolds numbers, bed shear stress) and the ecological stress indices.

Deviations from ML_STRATEGY.md section 3
----------------------------------------
Three corrections, each deliberate:

1. **Lagged and rolling features are not produced.** They assume a continuous
   hourly chronology. The dataset is a designed experiment over flow space
   (see ``docs/ARCHITECTURE.md``), where rows are independent samples and a
   "previous hour" does not exist. Producing them would invent autocorrelation.

2. **The thermal stress formula in the spec contradicts its own description.**
   ``1 / (1 + exp(-a * (temp - optimal)))`` is a monotonic sigmoid: it rises with
   temperature and never penalises cold water. The accompanying prose asks for a
   "penalty for temp > 25C or < 10C". The prose is ecologically right, so this
   module implements a two-sided response about the optimum.

3. **The oxygen stress formula is unbounded.** ``exp(-b * (optimal_do - do))``
   exceeds 1 whenever DO is above the optimum and grows without limit, so it
   cannot be a stress index. Implemented here as a bounded, saturating deficit
   response instead.

Ranges quoted in section 3.1 were guesses made for the Yodo. Use
:func:`observed_ranges` to derive bounds from the monitoring record instead;
Ayase discharge, for instance, is 0.17-64 m3/s against the spec's 50-500.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("data.preprocessor")

# Physical constants
GRAVITY = 9.80665  # m/s^2
WATER_DENSITY = 1000.0  # kg/m^3

# Ecological reference points. Broad cool-water assemblage typical of a lowland
# Kanto river; revisit if the target taxa are narrowed.
OPTIMAL_TEMP_C = 17.5
TEMP_TOLERANCE_C = 7.5  # stress reaches ~0.5 at optimum +/- this
CRITICAL_DO = 2.0  # mg/L - acute mortality threshold for most fish
AMPLE_DO = 8.0  # mg/L - no meaningful oxygen stress at or above this


# ---------------------------------------------------------------------------
# Physical relationships
# ---------------------------------------------------------------------------


def do_saturation(temperature_c, elevation_m=0.0):
    """Dissolved oxygen at saturation in mg/L for fresh water.

    Benson-Krause equation as given in APHA Standard Methods 4500-O, with an
    exponential pressure correction for elevation. Valid 0-40 C.

    This replaces the spec's fixed ``optimal_do = 10 mg/L``: saturation is
    strongly temperature dependent (14.6 mg/L at 0 C, 8.3 at 25 C), so a constant
    reference would misattribute ordinary summer warming as oxygen depletion.
    """
    t = np.asarray(temperature_c, dtype=np.float64)
    kelvin = t + 273.15

    ln_sat = (
        -139.34411
        + 1.575701e5 / kelvin
        - 6.642308e7 / kelvin**2
        + 1.243800e10 / kelvin**3
        - 8.621949e11 / kelvin**4
    )
    sat = np.exp(ln_sat)

    # Barometric pressure falls roughly exponentially with a scale height ~8.43 km.
    sat = sat * np.exp(-np.asarray(elevation_m, dtype=np.float64) / 8434.0)

    return np.where((t < 0.0) | (t > 40.0), np.nan, sat)


def kinematic_viscosity(temperature_c):
    """Kinematic viscosity of water in m2/s.

    Temperature dependent, because it varies by roughly a factor of two across
    the 5-32 C range observed on the Ayase and feeds straight into Reynolds.
    """
    t = np.asarray(temperature_c, dtype=np.float64)
    return 1.79e-6 / (1.0 + 0.03368 * t + 0.000221 * t**2)


def froude_number(velocity, depth):
    """Fr = v / sqrt(g*d). Below 1 is subcritical, above 1 supercritical."""
    v = np.asarray(velocity, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        fr = v / np.sqrt(GRAVITY * d)
    return np.where(d > 0, fr, np.nan)


def reynolds_number(velocity, depth, temperature_c=15.0):
    """Re = v*R/nu, taking hydraulic radius as depth (wide-channel approximation).

    Valid where width greatly exceeds depth, which holds for the Ayase: sections
    run tens of metres wide at a few metres deep.
    """
    v = np.asarray(velocity, dtype=np.float64)
    d = np.asarray(depth, dtype=np.float64)
    nu = kinematic_viscosity(temperature_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        re = np.abs(v) * d / nu
    return np.where(d > 0, re, np.nan)


def bed_shear_stress(depth, slope):
    """tau = rho*g*R*S in Pa, hydraulic radius approximated by depth."""
    d = np.asarray(depth, dtype=np.float64)
    s = np.asarray(slope, dtype=np.float64)
    tau = WATER_DENSITY * GRAVITY * d * s
    return np.where((d >= 0) & (s >= 0), tau, np.nan)


# ---------------------------------------------------------------------------
# Ecological stress indices - all in [0, 1], 0 = no stress
# ---------------------------------------------------------------------------


def thermal_stress(temperature_c, optimal=OPTIMAL_TEMP_C, tolerance=TEMP_TOLERANCE_C):
    """Two-sided thermal stress: 0 at the optimum, approaching 1 at extremes.

    Gaussian about the optimum, so cold and warm departures are both penalised.
    See the module docstring for why this departs from the spec's sigmoid.
    """
    t = np.asarray(temperature_c, dtype=np.float64)
    return 1.0 - np.exp(-(((t - optimal) / tolerance) ** 2))


def oxygen_stress(dissolved_oxygen, critical=CRITICAL_DO, ample=AMPLE_DO):
    """Oxygen stress: 1 at or below ``critical``, 0 at or above ``ample``.

    Between the two the response is squared rather than linear, because the
    ecological cost of losing oxygen accelerates as levels fall.
    """
    do = np.asarray(dissolved_oxygen, dtype=np.float64)
    span = ample - critical
    if span <= 0:
        raise ValueError("ample DO must exceed critical DO")
    scaled = np.clip((ample - do) / span, 0.0, 1.0)
    return np.where(np.isnan(do), np.nan, scaled**2)


def combined_stress(thermal, oxygen):
    """Euclidean combination of the two stresses, normalised back to [0, 1]."""
    t = np.asarray(thermal, dtype=np.float64)
    o = np.asarray(oxygen, dtype=np.float64)
    return np.sqrt(t**2 + o**2) / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Frame-level feature construction
# ---------------------------------------------------------------------------

DERIVED_FEATURES = [
    "do_saturation",
    "do_deficit",
    "do_saturation_pct",
    "froude_number",
    "reynolds_number",
    "shear_stress",
    "thermal_stress_index",
    "oxygen_stress_index",
    "combined_stress_index",
    "temp_x_do",
    "flow_x_sediment",
    "discharge_x_temp",
]


def add_derived_features(
    frame: pd.DataFrame,
    temperature: str = "water_temp",
    oxygen: str = "dissolved_oxygen",
    depth: str = "depth",
    velocity: str = "velocity",
    discharge: str = "discharge",
    sediment: str = "suspended_solids",
    slope: str | float = 1e-4,
    elevation_m: float = 0.0,
) -> pd.DataFrame:
    """Append derived features to a frame, computing whatever the inputs allow.

    Columns whose inputs are absent are skipped with a debug note rather than
    raising, so the same function serves the observation-only frame and the
    fuller simulated one.
    """
    out = frame.copy()
    has = out.columns.__contains__

    if has(temperature):
        temp = out[temperature]
        sat = do_saturation(temp, elevation_m)
        out["do_saturation"] = sat
        out["thermal_stress_index"] = thermal_stress(temp)

        if has(oxygen):
            out["do_deficit"] = sat - out[oxygen]
            with np.errstate(divide="ignore", invalid="ignore"):
                out["do_saturation_pct"] = 100.0 * out[oxygen] / sat
            out["oxygen_stress_index"] = oxygen_stress(out[oxygen])
            out["combined_stress_index"] = combined_stress(
                out["thermal_stress_index"], out["oxygen_stress_index"]
            )
            out["temp_x_do"] = temp * out[oxygen]

        if has(discharge):
            out["discharge_x_temp"] = out[discharge] * temp
    else:
        log.debug("no %s column; thermal and oxygen features skipped", temperature)

    if has(depth) and has(velocity):
        out["froude_number"] = froude_number(out[velocity], out[depth])
        out["reynolds_number"] = reynolds_number(
            out[velocity], out[depth], out[temperature] if has(temperature) else 15.0
        )

    if has(depth):
        slope_values = out[slope] if isinstance(slope, str) and has(slope) else slope
        out["shear_stress"] = bed_shear_stress(out[depth], slope_values)

    if has(discharge) and has(sediment):
        out["flow_x_sediment"] = out[discharge] * out[sediment]

    added = [c for c in DERIVED_FEATURES if c in out.columns and c not in frame.columns]
    log.info("added %d derived feature(s): %s", len(added), ", ".join(added))
    out.attrs = dict(frame.attrs)
    return out


def observed_ranges(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Empirical bounds per variable, for designing the simulation sweep.

    Quantiles rather than min/max, so a single fouled sample cannot stretch the
    design space. Returns count, mean, std and the quantile bounds.
    """
    numeric = frame.select_dtypes(include=[np.number])
    if columns:
        missing = set(columns) - set(numeric.columns)
        if missing:
            log.warning("not numeric or not present: %s", sorted(missing))
        numeric = numeric[[c for c in columns if c in numeric.columns]]

    stats = pd.DataFrame(
        {
            "count": numeric.count(),
            "mean": numeric.mean(),
            "std": numeric.std(),
            "min": numeric.min(),
            f"q{lower:g}": numeric.quantile(lower),
            f"q{upper:g}": numeric.quantile(upper),
            "max": numeric.max(),
        }
    )
    return stats.dropna(subset=["count"]).query("count > 0")
