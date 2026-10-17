"""Tests for land use mesh selection and listing parsing."""
# ruff: noqa: E501

from __future__ import annotations

import numpy as np

from aquanexus.data import landuse

HTML = """
<tr><td id="a5339-1">5339</td><td class="txtCenter">世界測地系</td>
<td class="txtCenter">平成28年</td><td class="txtCenter">12.50MB</td>
<td class="txtCenter">L03-b-16_5339-jgd_GML.zip</td><td>
<a onclick="javascript:DownLd('12.50MB','L03-b-16_5339-jgd_GML.zip','/ksj/gml/data/L03-b/L03-b-16/L03-b-16_5339-jgd_GML.zip' ,this);"></a></td></tr>
<tr><td id="a5339-2">5339</td><td class="txtCenter">日本測地系</td>
<td class="txtCenter">平成28年</td><td class="txtCenter">12.40MB</td>
<td class="txtCenter">L03-b-16_5339-tky_GML.zip</td><td>
<a onclick="javascript:DownLd('12.40MB','L03-b-16_5339-tky_GML.zip','/ksj/gml/data/L03-b/L03-b-16/L03-b-16_5339-tky_GML.zip' ,this);"></a></td></tr>
<tr><td id="a5340-3">5340</td><td class="txtCenter">世界測地系</td>
<td class="txtCenter">平成26年</td><td class="txtCenter">3.00MB</td>
<td class="txtCenter">L03-b-14_5340-jgd_GML.zip</td><td>
<a onclick="javascript:DownLd('3.00MB','L03-b-14_5340-jgd_GML.zip','/ksj/gml/data/L03-b/L03-b-14/L03-b-14_5340-jgd_GML.zip' ,this);"></a></td></tr>
"""


def test_mesh_code_matches_the_published_scheme():
    # Tokyo station: 35.68 N, 139.77 E lies in 1st mesh 5339
    assert landuse.mesh1_code(35.68, 139.77) == 5339
    assert landuse.mesh1_code(np.array([43.06]), np.array([141.35]))[0] == 6441


def test_listing_parses_and_selection_keeps_only_the_wanted_year_datum_and_codes():
    listing = landuse.parse_listing(HTML)
    assert len(listing) == 3
    chosen = landuse.select_files(listing, [5339, 5340])
    assert chosen["file"].tolist() == ["L03-b-16_5339-jgd_GML.zip"]
    assert chosen.iloc[0]["path"].startswith("/ksj/gml/data/L03-b/L03-b-16/")


def _tile(codes):
    return landuse.Tile(np.array(codes, dtype=np.int16), 35.0, 139.0, 35.0 + 0.04, 139.0 + 0.04)


def test_shares_are_over_land_cells_and_sea_is_reported_separately():
    # 40 x 40 grid: west half forest (500), east half sea (1500)
    grid = np.full((40, 40), 500)
    grid[:, 20:] = 1500
    tile = _tile(grid)
    near_coast = landuse.window_shares(tile, 35.02, 139.02, 3.0)
    assert near_coast["lu_forest"] == 1.0  # all land is forest, sea is not in the denominator
    assert 0.0 < near_coast["lu_sea"] < 1.0
    inland = landuse.window_shares(tile, 35.02, 139.005, 0.3)
    assert inland["lu_sea"] == 0.0


def test_row_zero_is_the_northern_edge():
    grid = np.full((40, 40), 500)
    grid[:20, :] = 700  # northern half urban
    tile = _tile(grid)
    north = landuse.window_shares(tile, 35.035, 139.02, 0.4)
    south = landuse.window_shares(tile, 35.005, 139.02, 0.4)
    assert north["lu_urban"] == 1.0 and south["lu_forest"] == 1.0


def test_read_tile_round_trips_a_gml_zip(tmp_path):
    import zipfile

    xml = (
        "<ksj:Dataset><gml:lowerCorner>35.0 139.0</gml:lowerCorner>"
        "<gml:upperCorner>35.5 140.0</gml:upperCorner>"
        "<gml:high>2 1</gml:high><gml:tupleList>\n 0100 0200 0500 0700 0901 1500 \n</gml:tupleList>"
    )
    path = tmp_path / "L03-b-16_5339-jgd_GML.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("L03-b-16_5339.xml", xml)
        archive.writestr("KS-META-L03-b-16_5339.xml", "<meta/>")
    tile = landuse.read_tile(path)
    assert tile.codes.shape == (2, 3)
    assert tile.codes[1, 2] == 1500 and tile.north == 35.5 and tile.west == 139.0
