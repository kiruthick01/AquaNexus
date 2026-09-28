"""Tests for the Phase 4 remote-sensing infrastructure.

Everything here runs against synthetic band arrays and a synthetic raster
grid - see ``src/aquanexus/remote_sensing.py`` for why no real satellite
imagery is used: no citable station coordinates exist yet to fetch imagery
for.
"""

import numpy as np
import pandas as pd
import pytest

from aquanexus.remote_sensing import (
    STATION_COORDINATES,
    MissingStationCoordinatesError,
    RasterGrid,
    buffer_statistics,
    circular_buffer_mask,
    cloud_free_mask,
    mndwi,
    ndvi,
    ndwi,
    nearest_acquisition,
    require_station_coordinates,
)

# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------


def test_ndvi_matches_hand_computed_value():
    nir = np.array([0.5, 0.8])
    red = np.array([0.1, 0.2])
    result = ndvi(nir, red)
    expected = (nir - red) / (nir + red)
    np.testing.assert_allclose(result, expected)


def test_ndvi_is_nan_not_error_at_zero_denominator():
    result = ndvi(np.array([0.0]), np.array([0.0]))
    assert np.isnan(result[0])


def test_ndwi_uses_green_minus_nir_sign_convention():
    green, nir = np.array([0.6]), np.array([0.2])
    result = ndwi(nir, green)
    expected = (green - nir) / (green + nir)
    np.testing.assert_allclose(result, expected)


def test_mndwi_matches_hand_computed_value():
    green, swir = np.array([0.5]), np.array([0.1])
    result = mndwi(green, swir)
    expected = (green - swir) / (green + swir)
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# Cloud filtering
# ---------------------------------------------------------------------------


def test_cloud_free_mask_keeps_only_clear_classes():
    scl = np.array([0, 3, 4, 6, 8, 9, 10, 11])
    mask = cloud_free_mask(scl)
    # 4 (vegetation) and 6 (water) are clear; nothing else in this array is.
    np.testing.assert_array_equal(mask, [False, False, True, True, False,
                                         False, False, False])


# ---------------------------------------------------------------------------
# Spatial buffers
# ---------------------------------------------------------------------------


def test_circular_buffer_mask_includes_center_excludes_far_corner():
    # 11x11 grid, 10 m pixels, origin at (0, 110) so row 0 is the north edge.
    grid = RasterGrid(origin_x=0.0, origin_y=110.0, pixel_size=10.0)
    shape = (11, 11)
    center = (55.0, 55.0)  # the middle of the grid
    mask = circular_buffer_mask(grid, shape, center, radius_m=20.0)
    # The centre pixel (row 5, col 5) must be included; the far corner must not.
    assert mask[5, 5]
    assert not mask[0, 0]
    assert not mask[10, 10]


def test_buffer_statistics_reports_mean_and_valid_fraction():
    grid = RasterGrid(origin_x=0.0, origin_y=50.0, pixel_size=10.0)
    band = np.full((5, 5), 2.0)
    band[0, 0] = np.nan  # one pixel unreadable
    center = (25.0, 25.0)
    stats = buffer_statistics(band, grid, center, radius_m=100.0)  # whole grid
    assert stats["n_pixels"] == 25
    assert stats["n_valid"] == 24
    assert stats["valid_fraction"] == pytest.approx(24 / 25)
    assert stats["mean"] == pytest.approx(2.0)


def test_buffer_statistics_all_cloud_returns_nan_not_crash():
    grid = RasterGrid(origin_x=0.0, origin_y=50.0, pixel_size=10.0)
    band = np.full((5, 5), 2.0)
    valid_mask = np.zeros((5, 5), dtype=bool)  # everything cloud-masked out
    stats = buffer_statistics(band, grid, (25.0, 25.0), radius_m=100.0,
                              valid_mask=valid_mask)
    assert stats["n_valid"] == 0
    assert np.isnan(stats["mean"])
    assert stats["valid_fraction"] == 0.0


# ---------------------------------------------------------------------------
# Temporal matching
# ---------------------------------------------------------------------------


def test_nearest_acquisition_within_tolerance():
    acquisitions = pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-01"])
    matched, gap = nearest_acquisition(acquisitions, "2024-01-08", max_gap_days=5)
    assert matched == pd.Timestamp("2024-01-10")
    assert gap == 2.0


def test_nearest_acquisition_rejects_out_of_tolerance():
    acquisitions = pd.to_datetime(["2024-01-01", "2024-03-01"])
    matched, gap = nearest_acquisition(acquisitions, "2024-01-20", max_gap_days=5)
    assert matched is None
    assert gap > 5


def test_nearest_acquisition_empty_series():
    matched, gap = nearest_acquisition([], "2024-01-01", max_gap_days=5)
    assert matched is None
    assert gap == float("inf")


# ---------------------------------------------------------------------------
# Station coordinates - the actual Phase 4 blocker
# ---------------------------------------------------------------------------


def test_station_coordinates_registry_is_currently_empty():
    # Documents the blocked state itself: if this ever fails, station
    # coordinates have been added and docs/ML_ROADMAP.md Phase 4's status
    # needs revisiting, not just this test.
    assert STATION_COORDINATES == {}


def test_require_station_coordinates_raises_for_real_ayase_stations():
    real_ayase_stations = ["52内匠橋", "54槐戸橋", "55畷橋", "57綾瀬川合流点前"]
    with pytest.raises(MissingStationCoordinatesError, match="内匠橋"):
        require_station_coordinates(real_ayase_stations)


def test_require_station_coordinates_succeeds_once_populated(monkeypatch):
    monkeypatch.setitem(STATION_COORDINATES, "test-station", (35.9, 139.6))
    result = require_station_coordinates(["test-station"])
    assert result == {"test-station": (35.9, 139.6)}
