"""Reader tests, including a round-trip against the geometry exporter.

The round-trip is the useful one: it proves the .g01 this package writes is
parseable as the format it claims to be, without needing HEC-RAS installed.
"""

import numpy as np
import pytest

from aquanexus.hecras.geometry import CrossSection, to_hecras_geometry
from aquanexus.hecras.reader import (
    find_projects,
    read_geometry,
    read_project,
)

PRJ = """Proj Title=Ayase Test
Description=Synthetic project for tests
Current Plan=p01
Geom File=g01
Geom File=g02
Flow File=f01
Unsteady File=u01
Plan File=p01
"""


def make_sections(n=4, points=12):
    out = []
    for k in range(n):
        station = np.linspace(0.0, 55.0, points)
        elevation = 6.0 + 0.05 * k + 0.004 * (station - 27.5) ** 2
        out.append(
            CrossSection(
                river_station=k * 500.0,
                station=station,
                elevation=elevation,
                origin=(0.0, 0.0),
                direction=(1.0, 0.0),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Project file
# ---------------------------------------------------------------------------


def test_read_project_fields(tmp_path):
    p = tmp_path / "ayase.prj"
    p.write_text(PRJ, encoding="utf-8")
    proj = read_project(p)

    assert proj.title == "Ayase Test"
    assert proj.description.startswith("Synthetic")
    assert proj.geometry_files == ["g01", "g02"]
    assert set(proj.flow_files) == {"f01", "u01"}
    assert proj.plan_files == ["p01"]
    assert proj.geometry_paths()[0] == tmp_path / "g01"


def test_read_project_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_project(tmp_path / "nope.prj")


def test_find_projects_skips_esri_projection_files(tmp_path):
    (tmp_path / "ras.prj").write_text(PRJ, encoding="utf-8")
    (tmp_path / "shapefile.prj").write_text(
        'PROJCS["JGD_2011_Japan_Zone_9",GEOGCS["GCS_JGD_2011"]]', encoding="utf-8"
    )
    found = find_projects(tmp_path)
    assert [f.name for f in found] == ["ras.prj"]


# ---------------------------------------------------------------------------
# Geometry round-trip
# ---------------------------------------------------------------------------


def test_geometry_roundtrip_preserves_values(tmp_path):
    original = make_sections(n=4, points=12)
    g01 = tmp_path / "model.g01"
    g01.write_text(to_hecras_geometry(original, river="Ayase", reach="Main"), encoding="utf-8")

    reaches = read_geometry(g01)
    assert len(reaches) == 1
    assert reaches[0].river == "Ayase"
    assert reaches[0].reach == "Main"
    assert len(reaches[0].sections) == len(original)

    by_rs = {s.river_station: s for s in reaches[0].sections}
    for src in original:
        got = by_rs[src.river_station]
        # Export writes 2 decimal places, so compare at that tolerance.
        assert np.allclose(got.station, src.station, atol=0.005)
        assert np.allclose(got.elevation, src.elevation, atol=0.005)


def test_roundtrip_preserves_point_counts(tmp_path):
    original = make_sections(n=3, points=17)  # not a multiple of the 5-pairs-per-line
    g01 = tmp_path / "odd.g01"
    g01.write_text(to_hecras_geometry(original), encoding="utf-8")
    sections = read_geometry(g01)[0].sections
    assert all(len(s.station) == 17 for s in sections)


def test_roundtrip_survives_wide_stations(tmp_path):
    """Stations past 4 digits still have to split cleanly at 8 characters."""
    xs = CrossSection(
        river_station=0.0,
        station=np.array([0.0, 1234.56, 9999.99]),
        elevation=np.array([12.34, 5.67, 11.0]),
        origin=(0.0, 0.0),
        direction=(1.0, 0.0),
    )
    g01 = tmp_path / "wide.g01"
    g01.write_text(to_hecras_geometry([xs]), encoding="utf-8")
    got = read_geometry(g01)[0].sections[0]
    assert np.allclose(got.station, xs.station, atol=0.005)


def test_reach_length(tmp_path):
    g01 = tmp_path / "r.g01"
    g01.write_text(to_hecras_geometry(make_sections(n=4)), encoding="utf-8")
    assert read_geometry(g01)[0].length == pytest.approx(1500.0)


def test_read_geometry_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_geometry(tmp_path / "absent.g01")


def test_empty_geometry_yields_reach_without_sections(tmp_path):
    g01 = tmp_path / "empty.g01"
    g01.write_text(to_hecras_geometry([]), encoding="utf-8")
    reaches = read_geometry(g01)
    assert len(reaches) == 1
    assert reaches[0].sections == []
    assert reaches[0].length == 0.0


def test_sections_without_reach_header_are_still_captured(tmp_path):
    g01 = tmp_path / "headerless.g01"
    g01.write_text(
        "Geom Title=Orphan\n"
        "Type RM Length L Ch R = 1 ,100.00,,,\n"
        "#Sta/Elev= 2\n"
        f"{0.0:8.2f}{5.0:8.2f}{10.0:8.2f}{6.0:8.2f}\n"
        "#Mann= 3 , 0 , 0\n",
        encoding="utf-8",
    )
    reaches = read_geometry(g01)
    assert reaches[0].river == "Unknown"
    assert len(reaches[0].sections) == 1
