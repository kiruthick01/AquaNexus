"""Synthetic Habitat Suitability Index (HSI) labels.

HSI is not measured in the field for this project. Labels are generated from
ecological response curves over environmental state. **A model trained on these
labels is recovering a function written here, not learning ecology** - see the
README's "Scope and honest caveats".

What makes them defensible is that the response curves come from published
habitat requirements and that the resulting labels are checked against real
observations: :func:`falsification_report` verifies HSI actually degrades across
the temperature/oxygen gradient present in the monitoring record. If it does not,
the label function is wrong.

Habitat profiles
----------------
Suitability is meaningless without saying *for what*. ML_STRATEGY.md section 4.2
specifies coldwater fish (trout, char), which do not live in the Ayase - a
lowland Kanto river reaching 32.5 C in summer. Applying trout optima there scores
every summer sample at essentially zero and destroys the signal the model is
meant to learn. :data:`LOWLAND_WARMWATER` is therefore the default, describing the
eurythermal cyprinid assemblage (オイカワ, コイ, フナ) actually found in such
rivers. :data:`COLDWATER` retains the spec's parameters for comparison.

Corrections to ML_STRATEGY.md section 4.3
-----------------------------------------
1. **Dissolved oxygen used a two-sided Gaussian**, penalising high oxygen as
   heavily as low. At the spec's ``optimal=10, std=2``, an observed 12.85 mg/L
   scores 0.36 and a real 17.0 mg/L scores 0.002 - rated as harmful as 3.0 mg/L.
   Fish are not harmed by well-oxygenated water. Replaced with a plateau: full
   score at or above the optimum, with a separate penalty only for extreme
   supersaturation, which does cause gas-bubble trauma.

2. **The interaction penalty was a discontinuous step, and its branches were
   ordered so that worse conditions received a smaller penalty.** With the spec's
   ``if/elif``, 26 C with 3.5 mg/L DO takes the first branch (-0.15) while the
   milder 23 C with 3.5 mg/L takes the second (-0.20). Replaced with a smooth,
   monotonic term built from the product of thermal and oxygen stress.

3. **Sediment thresholds were set for a sediment-laden river.** The spec penalises
   above 50 mg/L; only 0.5% of Ayase samples exceed that, making the term nearly
   inert. Thresholds are profile parameters, tightened for a lowland river.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquanexus.data.preprocessor import do_saturation
from aquanexus.logger import get_logger

log = get_logger("data.synthetic")


@dataclass(frozen=True)
class HabitatProfile:
    """Habitat requirements for one species guild.

    ``*_tolerance`` values are the Gaussian width: score falls to ~0.61 one
    tolerance away from the optimum and ~0.14 at two.
    """

    name: str
    temp_optimal: float
    temp_tolerance: float
    do_optimal: float
    do_tolerance: float
    depth_optimal: float
    depth_tolerance: float
    velocity_optimal: float
    velocity_tolerance: float
    sediment_ok: float  # mg/L, no penalty at or below
    sediment_severe: float  # mg/L, score floors here
    # Acute lethality thresholds are profile parameters because tolerance genuinely
    # differs by guild: cyprinids survive oxygen levels that kill salmonids.
    do_lethal: float = 1.5
    do_safe: float = 3.5
    supersaturation_pct: float = 130.0  # % saturation before gas-bubble risk
    interaction_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.sediment_severe <= self.sediment_ok:
            raise ValueError("sediment_severe must exceed sediment_ok")
        for field in ("temp_tolerance", "do_tolerance", "depth_tolerance",
                      "velocity_tolerance"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")


#: Eurythermal cyprinid assemblage of lowland Kanto rivers - the Ayase default.
LOWLAND_WARMWATER = HabitatProfile(
    name="lowland_warmwater",
    temp_optimal=24.0,
    temp_tolerance=8.0,
    do_optimal=8.0,
    do_tolerance=3.0,
    depth_optimal=1.0,
    depth_tolerance=0.8,
    velocity_optimal=0.4,
    velocity_tolerance=0.35,
    sediment_ok=25.0,
    sediment_severe=200.0,
    do_lethal=1.5,
    do_safe=3.5,
)

#: ML_STRATEGY.md section 4.2 parameters. Retained for comparison; not
#: appropriate to the Ayase.
COLDWATER = HabitatProfile(
    name="coldwater",
    temp_optimal=15.0,
    temp_tolerance=4.0,
    do_optimal=10.0,
    do_tolerance=2.0,
    depth_optimal=1.2,
    depth_tolerance=0.4,
    velocity_optimal=0.5,
    velocity_tolerance=0.2,
    sediment_ok=50.0,
    sediment_severe=250.0,
    do_lethal=3.0,
    do_safe=5.0,
)

PROFILES = {p.name: p for p in (LOWLAND_WARMWATER, COLDWATER)}


# ---------------------------------------------------------------------------
# Response curves
# ---------------------------------------------------------------------------


def gaussian_score(value, optimal: float, tolerance: float):
    """Two-sided response: 1 at the optimum, decaying either side."""
    v = np.asarray(value, dtype=np.float64)
    return np.exp(-0.5 * ((v - optimal) / tolerance) ** 2)


def plateau_score(value, optimal: float, tolerance: float):
    """One-sided response: 1 at or above the optimum, Gaussian decay below.

    Used for dissolved oxygen, where more is not worse.
    """
    v = np.asarray(value, dtype=np.float64)
    return np.where(v >= optimal, 1.0, gaussian_score(v, optimal, tolerance))


def sediment_score(value, ok: float, severe: float):
    """1 at or below ``ok``, falling linearly to 0 at ``severe``."""
    v = np.asarray(value, dtype=np.float64)
    return np.clip(1.0 - (v - ok) / (severe - ok), 0.0, 1.0)


def supersaturation_score(dissolved_oxygen, temperature_c, limit_pct: float = 130.0):
    """Penalty for extreme oxygen supersaturation (gas-bubble trauma).

    Mild supersaturation is common and harmless - the Ayase record reaches 106%
    from spring algal photosynthesis - so this only bites well above saturation.
    """
    sat = do_saturation(temperature_c)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * np.asarray(dissolved_oxygen, dtype=np.float64) / sat
    excess = np.clip((pct - limit_pct) / 50.0, 0.0, 1.0)
    return np.where(np.isnan(pct), 1.0, 1.0 - excess)


def acute_survival(dissolved_oxygen, lethal: float = 1.5, safe: float = 3.5):
    """Survival multiplier for acutely low oxygen.

    Below roughly 2 mg/L most freshwater fish asphyxiate regardless of how
    favourable everything else is. Averaging factor scores cannot express that -
    an ideal depth and velocity would otherwise offset lethal water - so acute
    oxygen stress applies as a separate multiplier that drives HSI to zero.
    """
    do = np.asarray(dissolved_oxygen, dtype=np.float64)
    return np.clip((do - lethal) / (safe - lethal), 0.0, 1.0)


def interaction_penalty(temp_score, do_score, weight: float = 0.25):
    """Smooth synergistic penalty for simultaneous thermal and oxygen stress.

    Replaces the spec's discontinuous if/elif steps. Built from the product of
    the two stresses, so it is monotonic: worse conditions always cost more.
    """
    thermal_stress = 1.0 - np.asarray(temp_score, dtype=np.float64)
    oxygen_stress = 1.0 - np.asarray(do_score, dtype=np.float64)
    return weight * thermal_stress * oxygen_stress


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

FACTOR_COLUMNS = {
    "temperature": ("water_temp", "temperature"),
    "oxygen": ("dissolved_oxygen",),
    "depth": ("depth",),
    "velocity": ("velocity",),
    "sediment": ("suspended_solids", "suspended_sediment"),
}


def _resolve(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def habitat_suitability(
    frame: pd.DataFrame,
    profile: HabitatProfile | str = LOWLAND_WARMWATER,
    return_components: bool = False,
) -> pd.Series | pd.DataFrame:
    """Generate HSI in [0, 1] for each row.

    Factors present in the frame are scored and averaged; sediment and
    supersaturation apply as multipliers, and simultaneous thermal/oxygen stress
    adds a synergistic penalty. Missing factors are skipped rather than assumed,
    so an observation-only frame (temperature, oxygen, sediment) works without
    inventing depth and velocity.
    """
    if isinstance(profile, str):
        if profile not in PROFILES:
            raise KeyError(f"unknown profile {profile!r}; have {sorted(PROFILES)}")
        profile = PROFILES[profile]

    resolved = {k: _resolve(frame, v) for k, v in FACTOR_COLUMNS.items()}
    if resolved["temperature"] is None and resolved["oxygen"] is None:
        raise ValueError(
            "need at least temperature or dissolved oxygen to generate labels"
        )

    components: dict[str, np.ndarray] = {}
    n = len(frame)

    if col := resolved["temperature"]:
        components["temp_score"] = gaussian_score(
            frame[col], profile.temp_optimal, profile.temp_tolerance
        )
    if col := resolved["oxygen"]:
        components["do_score"] = plateau_score(
            frame[col], profile.do_optimal, profile.do_tolerance
        )
    if col := resolved["depth"]:
        components["depth_score"] = gaussian_score(
            frame[col], profile.depth_optimal, profile.depth_tolerance
        )
    if col := resolved["velocity"]:
        components["velocity_score"] = gaussian_score(
            frame[col], profile.velocity_optimal, profile.velocity_tolerance
        )

    # Geometric rather than arithmetic mean. The arithmetic mean lets favourable
    # factors compensate for a lethal one: at 33 C with 2.5 mg/L DO, optimal depth
    # and velocity scoring 1.0 each pulled HSI up to 0.58 - "moderate" habitat for
    # water a fish cannot breathe in. The geometric mean is standard in HSI
    # methodology precisely because any factor near zero drags the whole score down.
    stacked = np.column_stack(list(components.values()))
    base = np.exp(np.nanmean(np.log(np.clip(stacked, 1e-12, None)), axis=1))

    multiplier = np.ones(n)
    if col := resolved["oxygen"]:
        components["survival_score"] = acute_survival(
            frame[col], profile.do_lethal, profile.do_safe
        )
        multiplier = multiplier * components["survival_score"]

    if col := resolved["sediment"]:
        components["sediment_score"] = sediment_score(
            frame[col], profile.sediment_ok, profile.sediment_severe
        )
        multiplier = multiplier * components["sediment_score"]

    if resolved["temperature"] and resolved["oxygen"]:
        components["supersat_score"] = supersaturation_score(
            frame[resolved["oxygen"]],
            frame[resolved["temperature"]],
            profile.supersaturation_pct,
        )
        multiplier = multiplier * components["supersat_score"]

    penalty = np.zeros(n)
    if "temp_score" in components and "do_score" in components:
        penalty = interaction_penalty(
            components["temp_score"], components["do_score"], profile.interaction_weight
        )
        components["interaction_penalty"] = penalty

    hsi = np.clip(base * multiplier - penalty, 0.0, 1.0)

    log.info(
        "HSI [%s] over %d rows from %s: mean %.3f, sd %.3f",
        profile.name,
        n,
        ", ".join(k for k, v in resolved.items() if v),
        np.nanmean(hsi),
        np.nanstd(hsi),
    )

    if return_components:
        out = pd.DataFrame(components, index=frame.index)
        out["hsi"] = hsi
        return out
    return pd.Series(hsi, index=frame.index, name="hsi")


def classify(hsi) -> pd.Series:
    """Bin HSI into the interpretive bands of ML_STRATEGY.md section 1."""
    values = pd.Series(np.asarray(hsi, dtype=np.float64))
    return pd.cut(
        values,
        bins=[-0.001, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["unsuitable", "poor", "moderate", "good", "optimal"],
    )


# ---------------------------------------------------------------------------
# Falsification
# ---------------------------------------------------------------------------


def falsification_report(
    frame: pd.DataFrame,
    profile: HabitatProfile | str = LOWLAND_WARMWATER,
    temp_band: tuple[float, float] = (15.0, 30.0),
) -> dict:
    """Check generated labels against the real observed stress gradient.

    The labels are synthetic, but the monitoring record is not. If HSI does not
    fall as observed conditions worsen, the label function is wrong.

    Comparisons are confined to ``temp_band``. Without that the test is
    confounded and returns the wrong verdict: in a temperate river dissolved
    oxygen anti-correlates with temperature, so the most oxygen-rich samples are
    winter water. Judged for a warmwater guild those score poorly on temperature,
    which made an unconditioned DO comparison run backwards - DO<5 at 0.585
    against DO>=8 at 0.505 - while the label function was in fact behaving
    correctly. Comparing like with like is the whole point of the test.
    """
    hsi = habitat_suitability(frame, profile)
    report: dict = {"profile": getattr(profile, "name", profile), "n": len(frame)}

    temp_col = _resolve(frame, FACTOR_COLUMNS["temperature"])
    do_col = _resolve(frame, FACTOR_COLUMNS["oxygen"])
    checks: list[bool] = []

    in_band = (
        frame[temp_col].between(*temp_band)
        if temp_col
        else pd.Series(True, index=frame.index)
    )
    banded, hsi_banded = frame[in_band], hsi[in_band]
    report["n_in_band"] = int(in_band.sum())

    if do_col and len(banded) >= 4:
        low = banded[do_col] < 5.0
        high = banded[do_col] >= 8.0
        if low.any() and high.any():
            report["hsi_low_do"] = float(hsi_banded[low].mean())
            report["hsi_high_do"] = float(hsi_banded[high].mean())
            report["n_low_do"] = int(low.sum())
            report["n_high_do"] = int(high.sum())
            checks.append(report["hsi_low_do"] < report["hsi_high_do"])

        # Spearman rather than Pearson: the response is monotonic but saturating
        # (HSI plateaus once oxygen is ample), so a linear coefficient understates
        # an ordering that is in fact near-perfect. Diagnostic only - the group
        # contrasts are what decide the verdict.
        if hsi_banded.nunique() > 1 and banded[do_col].nunique() > 1:
            report["corr_hsi_do"] = float(
                banded[do_col].corr(hsi_banded, method="spearman")
            )
        else:
            report["corr_hsi_do"] = float("nan")

    if temp_col and do_col:
        stressed = (frame[temp_col] > 25.0) & (frame[do_col] < 5.0)
        benign = frame[temp_col].between(18.0, 25.0) & (frame[do_col] >= 7.0)
        if stressed.any() and benign.any():
            report["hsi_stressed"] = float(hsi[stressed].mean())
            report["hsi_benign"] = float(hsi[benign].mean())
            report["n_stressed"] = int(stressed.sum())
            report["n_benign"] = int(benign.sum())
            checks.append(report["hsi_stressed"] < report["hsi_benign"])

    report["passed"] = bool(checks) and all(checks)
    if not checks:
        report["reason"] = "insufficient contrast in the observations to test"

    log.info("falsification: %s", "PASS" if report["passed"] else "FAIL")
    return report
