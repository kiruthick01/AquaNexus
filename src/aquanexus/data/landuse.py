"""Land use around each station, from 国土数値情報 L03-b (土地利用細分メッシュ).

The dataset is a 100 m mesh of land use classes, published as one GML zip per 1st-order
mesh (a 40 km by 80 km tile). Dissolved oxygen in a river reflects what drains into it,
so each station gets the share of each land use class within fixed radii of it.

Source: 国土交通省 国土数値情報ダウンロードサイト, https://nlftp.mlit.go.jp/ksj/ .
Cite as: 出典 国土交通省 国土数値情報（土地利用細分メッシュ）を加工して作成.
"""

from __future__ import annotations

import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from aquanexus.logger import get_logger

log = get_logger("data.landuse")

SITE = "https://nlftp.mlit.go.jp"
LIST_URL = f"{SITE}/ksj/gml/datalist/KsjTmplt-L03-b.html"
DATUM, YEAR = "世界測地系", "平成28年"
USER_AGENT = "aquanexus-research/0.1 (+https://github.com/kiruthick01/AquaNexus)"

_ROW = re.compile(
    r"<td id=\"a(\d+)-\d+\">\d+</td>\s*<td class=\"txtCenter\">([^<]*)</td>\s*"
    r"<td class=\"txtCenter\">([^<]*)</td>\s*<td class=\"txtCenter\">([\d.]+)MB</td>\s*"
    r"<td class=\"txtCenter\">([^<]*)</td>.*?DownLd\('[^']*','[^']*','([^']*)'",
    re.S,
)


def mesh1_code(latitude, longitude):
    """1st-order mesh code (4 digits) containing each point: ``lat*1.5`` then ``lon-100``."""
    lat = np.floor(np.asarray(latitude, dtype=float) * 1.5).astype(int)
    lon = np.floor(np.asarray(longitude, dtype=float) - 100).astype(int)
    return lat * 100 + lon


def parse_listing(html: str) -> pd.DataFrame:
    """The download table: mesh ``code``, ``datum``, ``year``, ``mb``, ``file`` and ``path``."""
    rows = _ROW.findall(html)
    table = pd.DataFrame(rows, columns=["code", "datum", "year", "mb", "file", "path"])
    table["code"] = table["code"].astype(int)
    table["mb"] = table["mb"].astype(float)
    return table


def select_files(listing: pd.DataFrame, codes) -> pd.DataFrame:
    """The FY2016 world-datum files covering the given mesh codes."""
    wanted = listing[(listing["datum"] == DATUM) & (listing["year"] == YEAR)]
    return wanted[wanted["code"].isin(set(int(c) for c in codes))].reset_index(drop=True)


def _get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_listing() -> pd.DataFrame:
    return parse_listing(_get(LIST_URL).decode("utf-8", errors="replace"))


def download(files: pd.DataFrame, out_dir: Path | str, pause: float = 1.0) -> list[Path]:
    """Fetch each zip once, one at a time, skipping files that already pass a zip check.

    Requests are sequential with a pause so the public server is never loaded.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, row in files.iterrows():
        target = out_dir / row["file"]
        if target.exists() and _valid(target):
            paths.append(target)
            continue
        target.write_bytes(_get(SITE + row["path"]))
        if not _valid(target):
            target.unlink()
            raise RuntimeError(f"{row['file']} did not download as a valid zip")
        paths.append(target)
        log.info("%d/%d %s (%.1f MB)", i + 1, len(files), row["file"], row["mb"])
        time.sleep(pause)
    return paths


def _valid(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


# --- reading the mesh --------------------------------------------------------------

#: 土地利用種別 codes (L03-b v2.1) grouped into the classes used as features.
CLASSES = {
    "lu_paddy": (100,),
    "lu_farm": (200,),
    "lu_forest": (500,),
    "lu_barren": (600,),
    "lu_urban": (700, 901, 902),
    "lu_other": (1000, 1400, 1600),
    "lu_water": (1100,),
}
SEA, NO_DATA = 1500, 0
KM_PER_DEGREE = 111.2
_ENVELOPE = re.compile(
    r"<gml:lowerCorner>([\d.\-]+) ([\d.\-]+)</gml:lowerCorner>\s*"
    r"<gml:upperCorner>([\d.\-]+) ([\d.\-]+)</gml:upperCorner>"
)
_HIGH = re.compile(r"<gml:high>(\d+) (\d+)</gml:high>")


class Tile:
    """One 1st-order mesh: a grid of land use codes and its geographic bounds."""

    def __init__(self, codes: np.ndarray, south: float, west: float, north: float, east: float):
        self.codes = codes  # [row, col]; row 0 is the northern edge
        self.south, self.west, self.north, self.east = south, west, north, east
        self.rows, self.cols = codes.shape
        self.dlat = (north - south) / self.rows
        self.dlon = (east - west) / self.cols

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat < self.north and self.west <= lon < self.east


def read_tile(path: Path | str) -> Tile:
    """Parse a tile straight from its GML text; the codes are a plain space-separated list."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".xml") and "KS-META" not in n)
        text = archive.read(name).decode("utf-8", errors="replace")
    lat0, lon0, lat1, lon1 = (float(v) for v in _ENVELOPE.search(text).groups())
    high_x, high_y = (int(v) + 1 for v in _HIGH.search(text).groups())
    start = text.index("<gml:tupleList>") + len("<gml:tupleList>")
    body = text[start : text.index("</gml:tupleList>")]
    codes = np.array(body.split(), dtype=np.int16)
    if codes.size != high_x * high_y:
        raise ValueError(f"{path}: {codes.size} cells, expected {high_x * high_y}")
    return Tile(codes.reshape(high_y, high_x), lat0, lon0, lat1, lon1)


def window_shares(tile: Tile, lat: float, lon: float, radius_km: float) -> dict[str, float]:
    """Land use shares of the cells within ``radius_km`` of a point inside ``tile``.

    Shares are over land cells (sea and no-data cells are left out of the denominator),
    so a coastal station is not diluted by water that is not part of its catchment.
    ``lu_land_cells`` says how much land the share rests on; ``lu_sea`` is the sea fraction
    of all cells in the window.
    """
    row = (tile.north - lat) / tile.dlat
    col = (lon - tile.west) / tile.dlon
    half_rows = radius_km / (tile.dlat * KM_PER_DEGREE)
    half_cols = radius_km / (tile.dlon * KM_PER_DEGREE * np.cos(np.radians(lat)))
    r0, r1 = max(int(row - half_rows), 0), min(int(row + half_rows) + 1, tile.rows)
    c0, c1 = max(int(col - half_cols), 0), min(int(col + half_cols) + 1, tile.cols)
    rr, cc = np.mgrid[r0:r1, c0:c1]
    inside = ((rr + 0.5 - row) / half_rows) ** 2 + ((cc + 0.5 - col) / half_cols) ** 2 <= 1.0
    cells = tile.codes[r0:r1, c0:c1][inside]
    land = cells[(cells != SEA) & (cells != NO_DATA)]
    out = {
        "lu_land_cells": float(len(land)),
        "lu_sea": float((cells == SEA).mean()) if len(cells) else np.nan,
    }
    for name, codes in CLASSES.items():
        out[name] = float(np.isin(land, codes).mean()) if len(land) else np.nan
    return out


def station_landuse(
    stations: pd.DataFrame, tile_dir: Path | str, radii_km: tuple[float, ...] = (1.0, 5.0)
) -> pd.DataFrame:
    """Land use shares around each station, one column set per radius (suffix ``_<km>km``)."""
    tile_dir = Path(tile_dir)
    codes = mesh1_code(stations["latitude"], stations["longitude"])
    rows = []
    for code in sorted(set(codes.tolist())):
        path = tile_dir / f"L03-b-16_{code}-jgd_GML.zip"
        members = stations[codes == code]
        if not path.exists():
            continue
        tile = read_tile(path)
        for _, s in members.iterrows():
            row = {"station_id": s["station_id"]}
            for km in radii_km:
                shares = window_shares(tile, s["latitude"], s["longitude"], km)
                row |= {f"{k}_{km:g}km": v for k, v in shares.items()}
            rows.append(row)
    return pd.DataFrame(rows)
