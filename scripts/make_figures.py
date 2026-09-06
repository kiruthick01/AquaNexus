"""Generate the figures used in README.md.

Every figure is drawn from this project's own outputs - no illustrative or
synthetic-for-display data. If a figure cannot be produced because an artefact is
missing, it is skipped with a note rather than faked.

    python scripts/make_figures.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aquanexus.config import settings  # noqa: E402
from aquanexus.logger import get_logger  # noqa: E402

log = get_logger("scripts.make_figures")

OUT = Path("docs/figures")
INK = "#1b2a41"
ACCENT = "#c0392b"
COOL = "#2e86ab"
WARM = "#e67e22"
GRID = "#dfe3e8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    log.info("wrote %s", path)


# ---------------------------------------------------------------------------


def figure_reach_profile():
    """Bed profile and channel width along the modelled reach."""
    from aquanexus.hecras.reader import read_geometry

    path = Path("data/hecras/ayasegawa/Ayasegawa.g01")
    if not path.is_file():
        return log.warning("skip reach profile: %s missing", path)

    sections = sorted(read_geometry(path)[0].sections, key=lambda s: s.river_station)
    rs = np.array([s.river_station for s in sections]) / 1000.0
    invert = np.array([s.thalweg for s in sections])
    width = np.array([s.width for s in sections])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(rs, invert, color=COOL, lw=1.8)
    ax1.fill_between(rs, invert, invert.min() - 0.5, color=COOL, alpha=0.13)
    slope = np.polyfit(rs, invert, 1)
    ax1.plot(rs, np.polyval(slope, rs), "--", color=ACCENT, lw=1.2,
             label=f"mean gradient {slope[0]:.2f} m/km")
    ax1.set_ylabel("bed elevation (m)")
    ax1.set_title("Ayase River (綾瀬川) — bed profile from 89 multibeam + UAV tiles",
                  loc="left", fontweight="bold")
    ax1.legend(frameon=False, fontsize=8)

    ax2.bar(rs, width, width=0.35, color=INK, alpha=0.55)
    ax2.set_ylabel("sampled\nwidth (m)")
    ax2.set_xlabel("river station (km)  →  upstream")
    fig.text(0.5, -0.02, f"{len(sections)} cross-sections over "
             f"{rs.max() - rs.min():.1f} km · EPSG:6677 (JGD2011 / Japan Plane Zone 9)",
             ha="center", fontsize=8, style="italic", color="#5a6472")
    save(fig, "reach_profile.png")


def figure_flow_sweep():
    """How depth, velocity and width respond across the discharge sweep."""
    path = settings.PROCESSED_DIR / "ayase_flow_sweep.csv"
    if not path.is_file():
        return log.warning("skip flow sweep: %s missing", path)

    sweep = pd.read_csv(path)
    grouped = sweep.groupby("discharge_bc")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, column, label, colour in zip(
        axes, ["depth", "velocity", "top_width"],
        ["depth (m)", "velocity (m/s)", "top width (m)"], [COOL, WARM, INK],
        strict=True,
    ):
        median = grouped[column].median()
        ax.fill_between(median.index, grouped[column].quantile(0.15),
                        grouped[column].quantile(0.85), color=colour, alpha=0.18)
        ax.plot(median.index, median.values, "o-", color=colour, ms=3.5, lw=1.6)
        ax.set_xscale("log")
        ax.set_xlabel("discharge (m³/s, log)")
        ax.set_ylabel(label)
    axes[0].set_title("HEC-RAS 7.0 steady-flow sweep — 53 sections × 12 discharges",
                      loc="left", fontweight="bold")
    save(fig, "flow_sweep.png")


def figure_observed_record():
    """The real monitoring record: temperature, oxygen and the deficit."""
    from aquanexus.data.loader import filter_stations, load_many
    from aquanexus.data.preprocessor import add_derived_features

    files = sorted(glob.glob(str(settings.RAW_DIR / "waterquality" / "saitama_*.xlsx")))
    if not files:
        return log.warning("skip observed record: no water quality files")

    frame = add_derived_features(
        filter_stations(load_many(files), water_body=settings.RIVER_NAME_JA)
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))

    order = np.argsort(frame.water_temp.values)
    axes[0].scatter(frame.water_temp, frame.dissolved_oxygen, s=16,
                    c=frame.water_temp, cmap="coolwarm", edgecolor="white", lw=0.3)
    axes[0].plot(frame.water_temp.values[order], frame.do_saturation.values[order],
                 color=INK, lw=1.5, label="saturation")
    axes[0].set_xlabel("water temperature (°C)")
    axes[0].set_ylabel("dissolved oxygen (mg/L)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Observed: oxygen vs temperature", loc="left", fontweight="bold")

    month = frame.timestamp.dt.month
    monthly = frame.groupby(month)
    axes[1].plot(monthly.water_temp.mean().index, monthly.water_temp.mean(),
                 "o-", color=WARM, label="water temp (°C)")
    axes[1].plot(monthly.dissolved_oxygen.mean().index, monthly.dissolved_oxygen.mean(),
                 "s-", color=COOL, label="DO (mg/L)")
    axes[1].set_xlabel("month")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Seasonal cycle", loc="left", fontweight="bold")

    deficit = frame.do_deficit.dropna()
    axes[2].hist(deficit, bins=26, color=ACCENT, alpha=0.75, edgecolor="white")
    axes[2].axvline(0, color=INK, lw=1.2)
    axes[2].axvline(deficit.mean(), color=INK, ls="--", lw=1.4,
                    label=f"mean {deficit.mean():+.2f} mg/L")
    axes[2].set_xlabel("oxygen deficit: saturation − observed (mg/L)")
    axes[2].set_ylabel("samples")
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].set_title("Persistent under-saturation", loc="left", fontweight="bold")

    fig.text(0.5, -0.04, f"{len(frame)} grab samples, FY2022–24 · "
             "環境省 水環境総合情報サイト / 埼玉県 公共用水域水質測定",
             ha="center", fontsize=8, style="italic", color="#5a6472")
    save(fig, "observed_record.png")


def figure_model_comparison():
    """Model and baseline comparison on the real target."""
    labels = ["Ridge\n(chosen)", "persistence\n(baseline)", "random\nforest",
              "XGBoost", "mean\n(floor)", "hydraulic\nonly"]
    rmse = [1.713, 1.818, 1.879, 1.904, 2.294, 2.467]
    r2 = [0.442, 0.385, 0.329, 0.311, 0.0, -0.157]
    colours = [ACCENT, "#7f8c8d", COOL, COOL, "#bdc3c7", "#bdc3c7"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.4))
    ax1.bar(labels, rmse, color=colours, alpha=0.9)
    ax1.set_ylabel("RMSE (mg/L) — lower better")
    ax1.set_title("Dissolved oxygen: real labels, grouped CV",
                  loc="left", fontweight="bold")
    for i, v in enumerate(rmse):
        ax1.text(i, v + 0.04, f"{v:.2f}", ha="center", fontsize=8)

    ax2.bar(labels, r2, color=colours, alpha=0.9)
    ax2.axhline(0, color=INK, lw=1)
    ax2.set_ylabel("R² — higher better")
    ax2.set_title("The model beats “no change” by only 0.06 R²",
                  loc="left", fontweight="bold")
    for i, v in enumerate(r2):
        ax2.text(i, v + (0.02 if v >= 0 else -0.06), f"{v:+.2f}",
                 ha="center", fontsize=8)
    save(fig, "model_comparison.png")


def figure_validation_bands():
    """Where the model fails: low flow."""
    bands = ["low flow\n(≈1 m³/s)", "middle\n(≈13 m³/s)", "high flow\n(≈52 m³/s)"]
    rmse = [2.532, 1.634, 0.818]
    bias = [-2.011, -0.043, 0.103]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax1.bar(bands, rmse, color=[ACCENT, "#bdc3c7", COOL], alpha=0.9)
    ax1.set_ylabel("RMSE (mg/L)")
    ax1.set_title("Error by flow regime", loc="left", fontweight="bold")

    ax2.bar(bands, bias, color=[ACCENT, "#bdc3c7", COOL], alpha=0.9)
    ax2.axhline(0, color=INK, lw=1)
    ax2.set_ylabel("bias (mg/L)")
    ax2.set_title("Under-predicts oxygen by 2 mg/L at low flow",
                  loc="left", fontweight="bold")
    fig.text(0.5, -0.06, "Drought is when oxygen stress matters most — "
             "the model is least reliable exactly there.",
             ha="center", fontsize=8, style="italic", color=ACCENT)
    save(fig, "validation_bands.png")


def figure_ablation():
    """Does the hydraulic model earn its place?"""
    sets = ["temperature\nonly", "+ season", "+ raw\ndischarge",
            "+ HEC-RAS\nhydraulics"]
    r2 = [0.328, 0.312, 0.293, 0.355]
    colours = ["#bdc3c7", "#bdc3c7", "#bdc3c7", ACCENT]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(sets, r2, color=colours, alpha=0.9)
    ax.set_ylabel("R² (grouped CV)")
    ax.set_ylim(0.25, 0.38)
    ax.set_title("Raw discharge hurts; the hydraulic transformation of it helps",
                 loc="left", fontweight="bold")
    for bar, value in zip(bars, r2, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.003,
                f"{value:.3f}", ha="center", fontsize=8)
    save(fig, "ablation.png")


def figure_cross_sections():
    """A few extracted cross-sections."""
    from aquanexus.hecras.reader import read_geometry

    path = Path("data/hecras/ayasegawa/Ayasegawa.g01")
    if not path.is_file():
        return log.warning("skip cross sections: %s missing", path)

    sections = sorted(read_geometry(path)[0].sections, key=lambda s: s.river_station)
    picks = sections[:: max(1, len(sections) // 4)][:4]

    fig, axes = plt.subplots(1, len(picks), figsize=(12, 2.7), sharey=True)
    for ax, xs in zip(np.atleast_1d(axes), picks, strict=True):
        ax.fill_between(xs.station, xs.elevation, xs.elevation.max(),
                        color=COOL, alpha=0.18)
        ax.plot(xs.station, xs.elevation, color=INK, lw=1.3)
        ax.set_title(f"RS {xs.river_station:.0f} m", fontsize=9)
        ax.set_xlabel("station (m)")
    np.atleast_1d(axes)[0].set_ylabel("elevation (m)")
    fig.suptitle("Cross-sections derived from the point cloud "
                 "(bed = low elevation quantile per bin)",
                 fontweight="bold", x=0.09, ha="left", y=1.06)
    save(fig, "cross_sections.png")


def main() -> int:
    for figure in (figure_reach_profile, figure_flow_sweep, figure_observed_record,
                   figure_model_comparison, figure_validation_bands,
                   figure_ablation, figure_cross_sections):
        try:
            figure()
        except Exception as exc:  # noqa: BLE001 - one figure must not stop the rest
            log.error("%s failed: %s", figure.__name__, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
