"""Cross-section extraction tests.

Built on a synthetic trapezoidal channel with known dimensions, so the tests
assert against the true geometry rather than against whatever the code currently
produces. No network or LAS fixture required.
"""

import numpy as np
import pytest

from aquanexus.hecras.geometry import (
    CrossSection,
    extract_cross_section,
    extract_sections,
    resample_centreline,
    smooth_centreline,
    to_hecras_geometry,
)

RNG = np.random.default_rng(42)


def synthetic_channel(
    length=1000.0,
    bed_width=20.0,
    depth=4.0,
    side_slope=3.0,
    bank_elev=10.0,
    pts_per_m2=4.0,
    noise=0.02,
    vegetation=True,
):
    """A straight trapezoidal channel running along +x, spanning y in [-60, 60].

    Bed at ``bank_elev - depth``, banks rising at 1:``side_slope``. Optionally
    adds a vegetation canopy above the banks, which is what the low-quantile
    binning has to reject.

    ``pts_per_m2`` is kept well below the real tiles' ~27 pts/m² to keep tests
    fast, but high enough that a 4 m strip still fills 1 m bins.
    """
    n = int(length * 120 * pts_per_m2)
    x = RNG.uniform(0, length, n)
    y = RNG.uniform(-60, 60, n)

    half_bed = bed_width / 2
    over = np.maximum(np.abs(y) - half_bed, 0.0)
    z = (bank_elev - depth) + over / side_slope
    z = np.minimum(z, bank_elev)
    z += RNG.normal(0, noise, n)

    pts = np.column_stack((x, y, z))

    if vegetation:
        m = n // 4
        vx = RNG.uniform(0, length, m)
        vy = np.where(RNG.random(m) < 0.5, RNG.uniform(-60, -half_bed - 8, m),
                      RNG.uniform(half_bed + 8, 60, m))
        vz = bank_elev + RNG.uniform(1.0, 9.0, m)
        pts = np.vstack((pts, np.column_stack((vx, vy, vz))))

    return pts


# ---------------------------------------------------------------------------
# Centreline
# ---------------------------------------------------------------------------


def test_resample_spacing_is_uniform():
    line = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0]])
    pts, tan = resample_centreline(line, spacing=25.0)
    steps = np.hypot(*np.diff(pts, axis=0).T)
    assert np.allclose(steps, 25.0)
    assert np.allclose(np.hypot(tan[:, 0], tan[:, 1]), 1.0)


def test_resample_follows_a_bend():
    line = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]])
    pts, tan = resample_centreline(line, spacing=50.0)
    # Last tangent should point along +y after the corner.
    assert tan[-1][1] == pytest.approx(1.0, abs=1e-6)


def test_resample_rejects_degenerate_input():
    with pytest.raises(ValueError):
        resample_centreline(np.array([[0.0, 0.0]]), 10.0)
    with pytest.raises(ValueError):
        resample_centreline(np.array([[0.0, 0.0], [1.0, 1.0]]), 0.0)
    with pytest.raises(ValueError):
        resample_centreline(np.zeros((3, 2)), 10.0)  # all identical -> zero length


def test_resample_tolerates_duplicate_vertices():
    line = np.array([[0.0, 0.0], [0.0, 0.0], [100.0, 0.0]])
    pts, _ = resample_centreline(line, spacing=50.0)
    assert len(pts) == 3


def test_smoothing_reduces_jitter_but_keeps_endpoints_near():
    straight = np.column_stack((np.arange(0, 200, 10.0), np.zeros(20)))
    noisy = straight + np.column_stack((np.zeros(20), RNG.normal(0, 3, 20)))
    smoothed = smooth_centreline(noisy, window=5)
    assert np.std(smoothed[:, 1]) < np.std(noisy[:, 1])
    assert len(smoothed) == len(noisy)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_recovers_known_bed_elevation_and_width():
    pts = synthetic_channel(bed_width=20.0, depth=4.0, bank_elev=10.0)
    xs = extract_cross_section(pts, origin=(500.0, 0.0), tangent=(1.0, 0.0),
                               half_width=60.0, bin_width=1.0)
    assert not xs.is_empty
    # Thalweg should sit at bank_elev - depth = 6.0
    assert xs.thalweg == pytest.approx(6.0, abs=0.15)
    # Flat bed should span roughly bed_width around the centre
    bed = xs.station[xs.elevation < 6.3]
    assert (bed.max() - bed.min()) == pytest.approx(20.0, abs=3.0)


def test_vegetation_does_not_lift_the_bed():
    """The whole point of the low quantile: canopy must not become terrain."""
    bare = synthetic_channel(vegetation=False)
    leafy = synthetic_channel(vegetation=True)
    a = extract_cross_section(bare, (500.0, 0.0), (1.0, 0.0))
    b = extract_cross_section(leafy, (500.0, 0.0), (1.0, 0.0))
    assert b.thalweg == pytest.approx(a.thalweg, abs=0.1)


def test_min_points_per_bin_filters_sparse_noise():
    pts = synthetic_channel()
    stray = np.array([[500.0, 58.0, -50.0]])  # single wild outlier far out
    xs = extract_cross_section(np.vstack((pts, stray)), (500.0, 0.0), (1.0, 0.0),
                               min_points_per_bin=3)
    assert xs.thalweg > 0.0  # the -50 m outlier was dropped


def test_stationing_increases_left_to_right():
    pts = synthetic_channel()
    xs = extract_cross_section(pts, (500.0, 0.0), (1.0, 0.0))
    assert np.all(np.diff(xs.station) > 0)
    assert xs.station.min() >= 0.0


def test_section_normal_is_perpendicular_to_flow():
    pts = synthetic_channel()
    xs = extract_cross_section(pts, (500.0, 0.0), (1.0, 0.0))
    # Channel is 40 m wide at the top; a section cut along flow instead of across
    # would sample a near-constant elevation.
    assert xs.elevation.max() - xs.elevation.min() > 2.0


def test_empty_when_far_from_cloud():
    pts = synthetic_channel()
    xs = extract_cross_section(pts, (5000.0, 5000.0), (1.0, 0.0))
    assert xs.is_empty
    assert np.isnan(xs.thalweg)
    assert xs.width == 0.0


def test_bed_quantile_bounds_are_validated():
    pts = synthetic_channel()
    with pytest.raises(ValueError):
        extract_cross_section(pts, (500.0, 0.0), (1.0, 0.0), bed_quantile=1.5)


def test_zero_tangent_rejected():
    with pytest.raises(ValueError):
        extract_cross_section(synthetic_channel(), (0.0, 0.0), (0.0, 0.0))


def test_bad_point_shape_rejected():
    with pytest.raises(ValueError):
        extract_cross_section(np.zeros((10, 2)), (0.0, 0.0), (1.0, 0.0))


def test_mismatched_arrays_rejected():
    with pytest.raises(ValueError):
        CrossSection(0.0, np.zeros(5), np.zeros(4), (0, 0), (1, 0))


# ---------------------------------------------------------------------------
# Reach-level
# ---------------------------------------------------------------------------


def test_extract_sections_spacing_and_count():
    pts = synthetic_channel(length=1000.0)
    line = np.column_stack((np.arange(0, 1001, 100.0), np.zeros(11)))
    sections = extract_sections(pts, line, spacing=200.0)
    assert len(sections) >= 5
    assert all(s.thalweg == pytest.approx(6.0, abs=0.3) for s in sections)
    stations = [s.river_station for s in sections]
    assert stations == sorted(stations)


def test_sections_past_the_cloud_are_dropped():
    pts = synthetic_channel(length=500.0)
    line = np.column_stack((np.arange(0, 2001, 100.0), np.zeros(21)))
    sections = extract_sections(pts, line, spacing=100.0)
    assert all(s.river_station <= 600.0 for s in sections)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_geometry_export_structure():
    pts = synthetic_channel(length=600.0)
    line = np.column_stack((np.arange(0, 601, 100.0), np.zeros(7)))
    sections = extract_sections(pts, line, spacing=200.0)
    text = to_hecras_geometry(sections, river="Ayase", reach="Main")

    # Names are padded to 16 characters. This is not cosmetic: without the
    # padding HEC-RAS 7.0 loads the file and reports zero rivers.
    assert f"River Reach={'Ayase':<16},{'Main':<16}" in text
    assert text.count("#Sta/Elev=") == len(sections)

    # HEC-RAS requires descending river station down the file.
    order = [
        float(ln.split(",")[1])
        for ln in text.splitlines()
        if ln.startswith("Type RM Length")
    ]
    assert order == sorted(order, reverse=True)


def test_geometry_export_pair_counts_match_header():
    pts = synthetic_channel(length=400.0)
    line = np.column_stack((np.arange(0, 401, 100.0), np.zeros(5)))
    sections = extract_sections(pts, line, spacing=200.0)
    text = to_hecras_geometry(sections)

    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if not ln.startswith("#Sta/Elev="):
            continue
        declared = int(ln.split("=")[1])
        pairs = 0
        for body in lines[i + 1 :]:
            if body.startswith("#Mann"):
                break
            pairs += len(body) // 16
        assert pairs == declared


def test_export_handles_empty_list():
    assert "River Reach=" in to_hecras_geometry([])


# ---------------------------------------------------------------------------
# HEC-RAS 7.0 format requirements
# ---------------------------------------------------------------------------
#
# Each element below was verified necessary against a real HEC-RAS 7.0 install
# through the COM controller. Omitting any of them makes HEC-RAS report zero
# rivers rather than raise an error, so these are regression guards.


def _sample_geometry():
    pts = synthetic_channel(length=600.0)
    line = np.column_stack((np.arange(0, 601, 100.0), np.zeros(7)))
    return to_hecras_geometry(extract_sections(pts, line, spacing=200.0))


def test_river_and_reach_names_are_padded_to_16_chars():
    for ln in _sample_geometry().splitlines():
        if ln.startswith("River Reach="):
            river, reach = ln[len("River Reach="):].split(",")
            assert len(river) == 16 and len(reach) == 16
            break
    else:
        raise AssertionError("no River Reach line emitted")


def test_reach_centreline_is_emitted():
    text = _sample_geometry()
    assert "Reach XY=" in text
    assert "Rch Text X Y=" in text
    assert "Reverse River Text=" in text


def test_each_section_carries_the_required_blocks():
    text = _sample_geometry()
    n = text.count("#Sta/Elev=")
    assert n > 0
    for marker in ("Node Last Edited Time=", "#Mann=", "Bank Sta=",
                   "XS Rating Curve=", "XS HTab Starting El and Incr=",
                   "XS HTab Horizontal Distribution=", "Exp/Cntr="):
        assert text.count(marker) == n, marker


def test_gis_cut_lines_are_opt_in():
    """A working reference project written by HEC-RAS 7.0 omits them entirely."""
    pts = synthetic_channel(length=600.0)
    line = np.column_stack((np.arange(0, 601, 100.0), np.zeros(7)))
    secs = extract_sections(pts, line, spacing=200.0)
    assert "XS GIS Cut Line=" not in to_hecras_geometry(secs)
    with_cuts = to_hecras_geometry(secs, gis_cut_lines=True)
    assert with_cuts.count("XS GIS Cut Line=") == len(secs)


def test_file_trailer_present():
    text = _sample_geometry()
    for marker in ("LCMann Time=", "LCMann Region Time=", "LCMann Table=",
                   "Chan Stop Cuts=", "Use User Specified Reach Order=",
                   "GIS Ratio Cuts To Invert=", "GIS Limit At Bridges=",
                   "Composite Channel Slope="):
        assert marker in text, marker


def test_htab_starting_elevation_sits_just_above_the_invert():
    pts = synthetic_channel(length=400.0)
    line = np.column_stack((np.arange(0, 401, 100.0), np.zeros(5)))
    secs = extract_sections(pts, line, spacing=200.0)
    text = to_hecras_geometry(secs)
    starts = [float(ln.split("=")[1].split(",")[0])
              for ln in text.splitlines() if ln.startswith("XS HTab Starting")]
    inverts = sorted((s.thalweg for s in secs), reverse=True)
    for start, invert in zip(starts, inverts, strict=True):
        assert invert < start < invert + 0.5


def test_bank_stations_lie_inside_the_section():
    text = _sample_geometry()
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if not ln.startswith("Bank Sta="):
            continue
        left, right = (float(v) for v in ln.split("=")[1].split(","))
        assert left < right
        # Recover this section's station range from the block above.
        for j in range(i, 0, -1):
            if lines[j].startswith("#Sta/Elev="):
                vals = []
                for body in lines[j + 1:]:
                    if body.startswith("#"):
                        break
                    vals += [float(body[k:k + 8]) for k in range(0, len(body), 8)
                             if body[k:k + 8].strip()]
                stations = vals[0::2]
                assert min(stations) <= left and right <= max(stations)
                break


def test_station_elevation_uses_8_char_fields_five_pairs_per_line():
    lines = _sample_geometry().splitlines()
    for i, ln in enumerate(lines):
        if not ln.startswith("#Sta/Elev="):
            continue
        for body in lines[i + 1:]:
            if body.startswith("#"):
                break
            assert len(body) % 8 == 0
            assert len(body) <= 80  # five pairs


def test_program_version_is_configurable():
    pts = synthetic_channel(length=400.0)
    line = np.column_stack((np.arange(0, 401, 100.0), np.zeros(5)))
    secs = extract_sections(pts, line, spacing=200.0)
    assert "Program Version=6.50" in to_hecras_geometry(secs, version="6.50")
