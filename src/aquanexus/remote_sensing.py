"""Satellite / remote-sensing feature extraction - infrastructure only.

**BLOCKED by missing station coordinates.** ``docs/ML_ROADMAP.md`` Phase 4 and
``docs/DATA_LIMITATIONS.md`` establish why: no latitude/longitude exists
anywhere in this repository for the Ayase or Naka water-quality monitoring
stations, only Japanese station names. The observation dates (2022-2025) are
comfortably within Sentinel-2 (2015-) and Landsat coverage, so temporal
compatibility is not the blocker - a citable coordinate source is.

Estimating coordinates from station names or approximate river-km position,
rather than sourcing them from a citable public record (e.g. the Saitama
prefecture monitoring program's own station registry), is explicitly out of
scope: it would introduce unverifiable spatial error into every downstream
buffer/NDVI feature, silently.

Everything below is generic feature-extraction machinery, tested against
synthetic raster arrays only (``tests/test_remote_sensing.py``), so that if
station coordinates are sourced and documented later, feature extraction can
proceed without rebuilding this layer.
:func:`require_station_coordinates` is the pipeline's entry guard - see
``scripts/phase4_remote_sensing_feasibility.py`` for it raising against the
real station list, which is the concrete evidence behind the "blocked" status
above, not just an inspection of the code.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("remote_sensing")


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------


def _safe_normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a - b) / (a + b)``, NaN where the denominator is zero rather than
    raising or silently emitting inf - a real possibility over water/shadow
    pixels where both bands can read near zero."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        result = np.where(denominator != 0, (a - b) / denominator, np.nan)
    return result


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index: ``(NIR - Red) / (NIR + Red)``."""
    return _safe_normalized_difference(nir, red)


def ndwi(nir: np.ndarray, green: np.ndarray) -> np.ndarray:
    """McFeeters' Normalized Difference Water Index: ``(Green - NIR) / (Green + NIR)``.

    Note the argument order matches the physical bands, not alphabetical
    convention: NDWI is green-minus-NIR, the opposite sign convention from
    NDVI's nir-minus-red.
    """
    return _safe_normalized_difference(green, nir)


def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Modified NDWI (Xu 2006): ``(Green - SWIR) / (Green + SWIR)``.

    Substitutes SWIR for NIR to suppress built-up-area false positives that
    plague plain NDWI - the reason it is offered as a separate index rather
    than a redundant one.
    """
    return _safe_normalized_difference(green, swir)


# ---------------------------------------------------------------------------
# Cloud filtering (Sentinel-2 Scene Classification Layer)
# ---------------------------------------------------------------------------

#: ESA's Sentinel-2 L2A Scene Classification (SCL) band codes.
#: 0 no data, 1 saturated/defective, 2 dark area, 3 cloud shadow,
#: 4 vegetation, 5 not vegetated, 6 water, 7 unclassified,
#: 8/9 cloud (medium/high probability), 10 thin cirrus, 11 snow.
SCL_CLEAR_CLASSES = (4, 5, 6, 7)
#: Dark areas (2) are deliberately excluded even though not strictly cloud:
#: they are frequently shadow or water-adjacent noise this project has no way
#: to distinguish without real imagery to inspect - a documented, conservative
#: choice, not an oversight.


def cloud_free_mask(scl: np.ndarray, clear_classes: tuple[int, ...] = SCL_CLEAR_CLASSES
                    ) -> np.ndarray:
    """Boolean mask of pixels whose SCL class is one of ``clear_classes``."""
    scl = np.asarray(scl)
    return np.isin(scl, clear_classes)


# ---------------------------------------------------------------------------
# Spatial buffers
# ---------------------------------------------------------------------------

#: Buffer radii under scientific consideration for a Sentinel-2-resolution
#: (10 m finest band) riparian study: 250 m (~78 pixels of area, a tight
#: near-bank zone), 500 m, 1 km, 2 km (catchment-scale context). None of these
#: is used by any function below with real data yet - they are the candidate
#: set `docs/REMOTE_SENSING.md` discusses once station coordinates exist.
CANDIDATE_BUFFERS_M = (250.0, 500.0, 1000.0, 2000.0)


@dataclass
class RasterGrid:
    """A minimal north-up affine grid: origin (top-left pixel centre is at
    ``origin_x``, ``origin_y``) plus a square pixel size in metres.

    Deliberately not a `rasterio` dataset - `rasterio` is an optional
    dependency (`pyproject.toml` ``geo`` extra) this project does not
    currently install, and the buffer geometry below needs nothing from it.
    If real imagery arrives via `rasterio`, its ``.transform`` maps onto this
    directly (``origin_x, _, pixel_size, origin_y, _, -pixel_size = transform``
    for a north-up, non-rotated raster).
    """

    origin_x: float
    origin_y: float
    pixel_size: float

    def pixel_centers(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = shape
        xs = self.origin_x + (np.arange(nx) + 0.5) * self.pixel_size
        ys = self.origin_y - (np.arange(ny) + 0.5) * self.pixel_size
        return np.meshgrid(xs, ys)


def circular_buffer_mask(grid: RasterGrid, shape: tuple[int, int],
                         center_xy: tuple[float, float], radius_m: float) -> np.ndarray:
    """Boolean mask of pixels whose centre lies within ``radius_m`` of ``center_xy``."""
    xs, ys = grid.pixel_centers(shape)
    cx, cy = center_xy
    distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return distance <= radius_m


def buffer_statistics(band: np.ndarray, grid: RasterGrid, center_xy: tuple[float, float],
                      radius_m: float, valid_mask: np.ndarray | None = None) -> dict:
    """Mean/std of ``band`` within a circular buffer, plus how much of it was usable.

    ``valid_fraction`` is reported explicitly rather than silently averaging
    over whatever passed the cloud mask - a buffer that is 90% cloud on the
    one clear-ish scene available is a materially weaker feature than one
    that is 90% clear, even if their means happen to match.
    """
    mask = circular_buffer_mask(grid, band.shape, center_xy, radius_m)
    n_total = int(mask.sum())
    if valid_mask is not None:
        mask = mask & valid_mask
    values = np.asarray(band, dtype=float)[mask]
    finite = values[np.isfinite(values)]
    n_valid = len(finite)
    return {
        "mean": float(np.mean(finite)) if n_valid else float("nan"),
        "std": float(np.std(finite)) if n_valid else float("nan"),
        "n_pixels": n_total,
        "n_valid": n_valid,
        "valid_fraction": n_valid / n_total if n_total else 0.0,
    }


# ---------------------------------------------------------------------------
# Temporal matching
# ---------------------------------------------------------------------------


def nearest_acquisition(acquisition_dates: Sequence, observation_date, max_gap_days: float
                        ) -> tuple[pd.Timestamp | None, float]:
    """Nearest acquisition date to a field-observation date, or ``(None, gap)``
    if the nearest one is farther than ``max_gap_days``.

    Same tolerance-based honesty as ``ml.forecasting``'s lag features: an
    acquisition 40 days from the field sample is not "the same-day scene" and
    must not be silently matched to it.
    """
    if len(acquisition_dates) == 0:
        return None, float("inf")
    dates = pd.to_datetime(pd.Series(acquisition_dates))
    observation = pd.Timestamp(observation_date)
    gaps = (dates - observation).abs().dt.days
    best = gaps.idxmin()
    best_gap = float(gaps.loc[best])
    if best_gap > max_gap_days:
        return None, best_gap
    return dates.loc[best], best_gap


# ---------------------------------------------------------------------------
# Station coordinates - the actual blocker
# ---------------------------------------------------------------------------


class MissingStationCoordinatesError(LookupError):
    """Raised by :func:`require_station_coordinates` for any station with no
    citable coordinate on record."""


#: No citable public coordinate source has been located for the Ayase or Naka
#: monitoring stations - see docs/DATA_LIMITATIONS.md Phase 4. Deliberately
#: empty rather than populated with estimated positions.
STATION_COORDINATES: dict[str, tuple[float, float]] = {}


def require_station_coordinates(station_names: Iterable[str]
                               ) -> dict[str, tuple[float, float]]:
    """Look up ``(lat, lon)`` for every name in ``station_names``, or raise
    naming exactly which ones are missing.

    This is the pipeline's entry guard: every function above is usable and
    tested today, but none of them can run on a real AquaNexus station until
    this stops raising for it.
    """
    names = list(station_names)
    missing = sorted({name for name in names if name not in STATION_COORDINATES})
    if missing:
        raise MissingStationCoordinatesError(
            f"no coordinates on record for {len(missing)}/{len(set(names))} station(s): "
            f"{missing}. See docs/DATA_LIMITATIONS.md Phase 4."
        )
    return {name: STATION_COORDINATES[name] for name in names}
