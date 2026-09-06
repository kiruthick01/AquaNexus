"""Build the modelling dataset - the "environmental state vectors".

This item is listed under Phase 1c in AQUANEXUS-PLAN.md but is really Phase 2
work: assembling it means deciding how simulated hydraulics join to observed
chemistry, which is an ML design choice rather than a data-pipeline step.

How the join works
------------------
Two sources describe the river, and neither is complete on its own:

* **HEC-RAS sweep** - depth, velocity, top width, flow area and energy slope for
  every cross-section at each of twelve discharges spanning the observed range.
  No water chemistry.
* **Monitoring record** - temperature, oxygen, nutrients and sediment, twelve
  grab samples per station per year, each carrying the discharge measured at the
  same moment. No hydraulics.

The discharge measured alongside each water sample is what ties them together.
For every observation, the sweep is interpolated to that observation's discharge
at every cross-section, producing one row per (section, observation).

What that assumes, and what it costs
------------------------------------
**Chemistry is treated as uniform along the reach on a given date.** It is not
exactly - the record has five stations over 27 km and they differ - but sampling
is monthly and only five points exist, so a per-section chemistry field would be
mostly interpolation. Holding chemistry constant while hydraulics vary is the
honest version of what the data supports: it lets the model learn how depth and
velocity modulate habitat under a given water quality, and no more.

The consequence is that the dataset has **group structure**. All 53 rows from one
observation share identical chemistry. A random train/test split would put
near-duplicate rows on both sides and report a score that is mostly memorisation.
:func:`build_state_vectors` therefore emits a ``group`` column, and splits must
be grouped on it - see ``aquanexus.ml.splits``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aquanexus.data.preprocessor import add_derived_features
from aquanexus.data.synthetic import LOWLAND_WARMWATER, HabitatProfile, habitat_suitability
from aquanexus.logger import get_logger

log = get_logger("data.dataset")

#: Hydraulic quantities interpolated from the sweep.
HYDRAULIC_COLUMNS = ("wse", "invert", "depth", "velocity",
                     "flow_area", "top_width", "energy_slope")

#: Chemistry carried across from the monitoring record.
OBSERVED_COLUMNS = ("water_temp", "air_temp", "dissolved_oxygen", "ph", "bod",
                    "cod", "suspended_solids", "nitrogen_total", "phosphorus_total")


def interpolate_hydraulics(sweep: pd.DataFrame, discharge: float) -> pd.DataFrame:
    """Interpolate every section's hydraulics to one discharge.

    The sweep is log-spaced because discharge spans two and a half orders of
    magnitude, so interpolation is done in log space too - linear interpolation
    between 0.17 and 73 m3/s would be dominated by the high end and badly
    misrepresent low flows, which is where habitat stress actually bites.

    Discharges outside the swept range are clamped rather than extrapolated.
    """
    if sweep.empty:
        raise ValueError("flow sweep is empty")

    grid = np.sort(sweep["discharge_bc"].unique())
    target = float(np.clip(discharge, grid.min(), grid.max()))
    log_grid, log_target = np.log(grid), np.log(target)

    rows = []
    for station, block in sweep.groupby("river_station", sort=True):
        block = block.sort_values("discharge_bc")
        row = {"river_station": station}
        for column in HYDRAULIC_COLUMNS:
            if column not in block.columns:
                continue
            values = block[column].to_numpy(dtype=float)
            row[column] = float(np.interp(log_target, log_grid, values))
        rows.append(row)

    return pd.DataFrame(rows)


def build_state_vectors(
    observations: pd.DataFrame,
    sweep: pd.DataFrame,
    profile: HabitatProfile = LOWLAND_WARMWATER,
    discharge_column: str = "discharge",
    require_discharge: bool = True,
) -> pd.DataFrame:
    """Join observed chemistry to simulated hydraulics and label the result.

    Returns one row per (cross-section, observation), carrying hydraulics
    interpolated to that observation's discharge, the observed chemistry, the
    derived features, and a synthetic HSI label.

    ``group`` identifies the source observation. Every split must be grouped on
    it; see the module docstring.
    """
    if discharge_column not in observations.columns:
        raise ValueError(f"observations have no '{discharge_column}' column")

    usable = observations.dropna(subset=[discharge_column]) if require_discharge \
        else observations
    if usable.empty:
        raise ValueError("no observations carry a discharge value")

    dropped = len(observations) - len(usable)
    if dropped:
        log.info("%d observation(s) without discharge were skipped", dropped)

    frames = []
    for group_id, (_, observation) in enumerate(usable.iterrows()):
        hydraulics = interpolate_hydraulics(sweep, float(observation[discharge_column]))
        hydraulics["group"] = group_id
        hydraulics["discharge"] = float(observation[discharge_column])

        for column in OBSERVED_COLUMNS:
            if column in usable.columns:
                hydraulics[column] = observation[column]
        for column in ("timestamp", "station", "water_body"):
            if column in usable.columns:
                hydraulics[column] = observation[column]

        frames.append(hydraulics)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = add_derived_features(dataset)
    dataset["hsi"] = habitat_suitability(dataset, profile)

    log.info(
        "state vectors: %d rows (%d sections x %d observations), %d feature column(s)",
        len(dataset),
        dataset["river_station"].nunique(),
        dataset["group"].nunique(),
        dataset.shape[1],
    )
    return dataset


def feature_columns(dataset: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Numeric columns usable as model inputs.

    Excludes identifiers, the target, and anything that is a *component* of the
    label function rather than an input to it. The stress indices and the
    suitability scores are computed from temperature and oxygen by the same code
    that writes the label, so feeding them back would be leakage dressed as a
    feature.
    """
    never = {
        "hsi", "group", "river_station",
        # Components of the label function - see the docstring.
        "thermal_stress_index", "oxygen_stress_index", "combined_stress_index",
    }
    never |= exclude or set()

    numeric = dataset.select_dtypes(include=[np.number])
    return [c for c in numeric.columns if c not in never and not c.endswith("_flag")]


# ---------------------------------------------------------------------------
# Observed water quality target
# ---------------------------------------------------------------------------

#: Physical and seasonal drivers of dissolved oxygen. Deliberately excludes the
#: other measured chemistry: BOD and nutrients are sampled from the same bottle,
#: so including them predicts one measurement from another rather than from the
#: river's physical state, and the hydraulics would stop mattering.
#: Physically possible range for dissolved oxygen in fresh water, used to clip
#: model output. Supersaturation to ~17 mg/L is real in this record.
DO_RANGE = (0.0, 25.0)

DO_FEATURES = (
    "water_temp", "air_temp", "discharge", "month_sin", "month_cos",
    "reach_depth", "reach_velocity", "reach_top_width", "reach_froude",
    "do_saturation",
)


def build_water_quality_dataset(
    observations: pd.DataFrame,
    sweep: pd.DataFrame,
    target: str = "dissolved_oxygen",
) -> pd.DataFrame:
    """Assemble a dataset for predicting an observed water quality variable.

    Unlike :func:`build_state_vectors`, this produces **one row per observation**
    - the real measurement is the label, so there is nothing to broadcast across
    cross-sections. Reach-averaged hydraulics at the observed discharge enter as
    features, which is how the hydraulic model contributes: dissolved oxygen is
    governed partly by reaeration, and reaeration is a function of depth and
    velocity.

    Station is kept as a column for grouping, not as a feature. With five
    stations and ~140 rows, a station identifier is close to a row identifier and
    a tree will use it to memorise.
    """
    usable = observations.dropna(subset=["discharge", target]).copy()
    if usable.empty:
        raise ValueError(f"no observations carry both discharge and {target}")

    rows = []
    for _, observation in usable.iterrows():
        discharge = float(observation["discharge"])
        hydraulics = interpolate_hydraulics(sweep, discharge)

        row = {
            "discharge": discharge,
            target: float(observation[target]),
            "reach_depth": float(hydraulics["depth"].mean()),
            "reach_velocity": float(hydraulics["velocity"].mean()),
            "reach_top_width": float(hydraulics["top_width"].mean()),
        }
        # Froude from reach means, guarding the shallow-water divide.
        depth = row["reach_depth"]
        row["reach_froude"] = (
            row["reach_velocity"] / np.sqrt(9.80665 * depth) if depth > 0 else np.nan
        )

        for column in ("water_temp", "air_temp", "station", "water_body", "timestamp"):
            if column in usable.columns:
                row[column] = observation[column]
        rows.append(row)

    dataset = pd.DataFrame(rows)

    # Season as a pair of cyclic terms, so December sits next to January rather
    # than eleven months away.
    month = pd.to_datetime(dataset["timestamp"]).dt.month
    dataset["month"] = month
    dataset["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    dataset["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    from aquanexus.data.preprocessor import do_saturation

    dataset["do_saturation"] = do_saturation(dataset["water_temp"])

    log.info("water quality dataset: %d observation(s), target %s, %d station(s)",
             len(dataset), target, dataset["station"].nunique()
             if "station" in dataset else 0)
    return dataset
