"""Project-writing tests.

Formats here were derived from a reference project HEC-RAS 7.0 wrote and
successfully computed, so these tests guard against drifting away from a shape
known to run.
"""

import numpy as np
import pytest

from aquanexus.hecras.geometry import CrossSection, bank_stations, to_hecras_geometry
from aquanexus.hecras.project import (
    SteadyFlowProfile,
    write_project,
    write_steady_flow,
)


def make_sections(n=3, points=21):
    """Sections with a V-shaped channel and a deliberately gappy station list."""
    out = []
    station = np.linspace(0.0, 100.0, points)
    station = np.delete(station, [7, 13])  # gaps, as dropped sparse bins create
    for k in range(n):
        elevation = 5.0 + 0.1 * k + 0.004 * (station - 50.0) ** 2
        out.append(
            CrossSection(
                river_station=100.0 + k * 100.0,
                station=station,
                elevation=elevation,
                origin=(1000.0 + k, 2000.0 + k),
                direction=(1.0, 0.0),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Bank stations
# ---------------------------------------------------------------------------


def test_bank_stations_are_real_station_values():
    """HEC-RAS refuses banks that are not present in the station/elevation list.

        - Left bank station not in station elevation data.

    An interpolated 'fraction of the width' guess lands in a gap and the whole
    run is rejected, so this is the regression that matters most here.
    """
    for xs in make_sections():
        left, right = bank_stations(xs)
        assert left in xs.station
        assert right in xs.station


def test_bank_stations_bracket_the_thalweg():
    xs = make_sections(1)[0]
    left, right = bank_stations(xs)
    thalweg_station = xs.station[int(np.argmin(xs.elevation))]
    assert left <= thalweg_station <= right
    assert left < right


def test_bank_stations_widen_with_height_fraction():
    xs = make_sections(1)[0]
    narrow = bank_stations(xs, height_fraction=0.2)
    wide = bank_stations(xs, height_fraction=0.9)
    assert (wide[1] - wide[0]) >= (narrow[1] - narrow[0])


def test_bank_stations_on_a_flat_profile_fall_back_to_the_ends():
    xs = CrossSection(0.0, np.arange(5.0), np.full(5, 7.0), (0, 0), (1, 0))
    assert bank_stations(xs) == (0.0, 4.0)


def test_bank_stations_empty_section():
    xs = CrossSection(0.0, np.empty(0), np.empty(0), (0, 0), (1, 0))
    assert bank_stations(xs) == (0.0, 0.0)


def test_emitted_bank_and_manning_stations_exist_in_the_data():
    text = to_hecras_geometry(make_sections())
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if not ln.startswith("#Sta/Elev="):
            continue
        values = []
        for body in lines[i + 1:]:
            if body.startswith("#"):
                break
            values += [float(body[k:k + 8]) for k in range(0, len(body), 8)
                       if body[k:k + 8].strip()]
        stations = set(np.round(values[0::2], 2))

        bank_line = next(x for x in lines[i:] if x.startswith("Bank Sta="))
        left, right = (round(float(v), 2) for v in bank_line.split("=")[1].split(","))
        assert left in stations and right in stations


# ---------------------------------------------------------------------------
# Flow file
# ---------------------------------------------------------------------------


def test_flow_file_river_unpadded_reach_padded(tmp_path):
    """The .f01 and .g01 disagree, and matching them is wrong.

    Geometry pads both names to 16; the flow file pads only the reach.
    """
    path = write_steady_flow(
        tmp_path / "t.f01", [SteadyFlowProfile("PF 1", 20.0)],
        river="Ayase", reach="Main", upstream_station=300,
    )
    line = next(x for x in path.read_text().splitlines()
                if x.startswith("River Rch & RM="))
    river, reach, station = line.split("=", 1)[1].split(",")
    assert river == "Ayase"        # not padded
    assert len(reach) == 16        # padded
    assert station.rstrip() == "300"


def test_flow_file_has_a_boundary_block_per_profile(tmp_path):
    profiles = [SteadyFlowProfile(f"P{i}", float(i)) for i in (1, 2, 3)]
    path = write_steady_flow(tmp_path / "t.f01", profiles, river="R", reach="M",
                             upstream_station=100)
    text = path.read_text()
    assert text.count("Boundary for River Rch & Prof#=") == 3
    assert "Number of Profiles= 3 " in text


def test_flow_values_share_one_line(tmp_path):
    path = write_steady_flow(
        tmp_path / "t.f01",
        [SteadyFlowProfile("a", 0.25), SteadyFlowProfile("b", 9.7),
         SteadyFlowProfile("c", 64.2)],
        river="R", reach="M", upstream_station=100,
    )
    lines = path.read_text().splitlines()
    values = lines[lines.index(next(x for x in lines if x.startswith("River Rch"))) + 1]
    assert [float(v) for v in values.split()] == [0.25, 9.7, 64.2]


def test_flow_file_rejects_empty_and_flat_slope(tmp_path):
    with pytest.raises(ValueError, match="at least one profile"):
        write_steady_flow(tmp_path / "a.f01", [], river="R", reach="M",
                          upstream_station=1)
    with pytest.raises(ValueError, match="downstream_slope"):
        write_steady_flow(tmp_path / "b.f01", [SteadyFlowProfile("p", 1.0)],
                          river="R", reach="M", upstream_station=1,
                          downstream_slope=0.0)


# ---------------------------------------------------------------------------
# Whole project
# ---------------------------------------------------------------------------


def test_write_project_emits_all_four_files(tmp_path):
    prj = write_project(
        tmp_path, "Ayase", to_hecras_geometry(make_sections()),
        [SteadyFlowProfile("PF 1", 9.7)],
        river="Ayase", reach="Main", upstream_station=300,
    )
    for ext in (".prj", ".g01", ".f01", ".p01"):
        assert prj.with_suffix(ext).is_file(), ext


def test_written_files_use_single_crlf(tmp_path):
    """Double CRLF is accepted by HEC-RAS, which then hangs on a dialog."""
    prj = write_project(
        tmp_path, "Ayase", to_hecras_geometry(make_sections()),
        [SteadyFlowProfile("PF 1", 9.7)],
        river="Ayase", reach="Main", upstream_station=300,
    )
    for ext in (".prj", ".g01", ".f01", ".p01"):
        raw = prj.with_suffix(ext).read_bytes()
        assert b"\r\r\n" not in raw, ext
        assert raw.count(b"\r\n") == raw.count(b"\n"), ext


def test_project_declares_si_units_and_current_plan(tmp_path):
    prj = write_project(
        tmp_path, "Ayase", to_hecras_geometry(make_sections()),
        [SteadyFlowProfile("PF 1", 9.7)],
        river="Ayase", reach="Main", upstream_station=300,
    )
    text = prj.read_text()
    assert "SI Units" in text          # else HEC-RAS silently assumes feet
    assert "Current Plan=p01" in text  # else geometry never loads


def test_plan_carries_the_full_settings_block(tmp_path):
    """A hand-minimised plan is rejected with an unhelpful generic message."""
    prj = write_project(
        tmp_path, "Ayase", to_hecras_geometry(make_sections()),
        [SteadyFlowProfile("PF 1", 9.7)],
        river="Ayase", reach="Main", upstream_station=300,
    )
    plan = prj.with_suffix(".p01").read_text()
    assert len(plan.splitlines()) > 150
    for key in ("Subcritical Flow", "Std Step Tol", "Num of Std Step Trials",
                "Geom File=g01", "Flow File=f01"):
        assert key in plan, key
