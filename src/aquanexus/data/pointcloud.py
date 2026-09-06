"""Discovery, download and loading of the Saitama river point cloud.

Saitama Prefecture publishes 3D point clouds for 59 managed rivers, acquired by
UAV drone *and narrow multibeam echosounder* - so unlike topographic LiDAR the
product includes the submerged bed. Data is CC BY 4.0; attribute
"埼玉県 河川点群データ" in anything derived from it.

The tile index is not published as a file. It is served as Mapbox vector tiles
for the prefecture's web map, where each polygon feature carries the tile id
(MESH_NO) and a direct download URL. :func:`fetch_tile_index` decodes those
tiles to recover the index.

See ``docs/DATA_SOURCES.md`` for the availability audit behind this module.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("data.pointcloud")

VECTOR_TILE_URL = (
    "https://gic-saitama.s3.ap-northeast-1.amazonaws.com"
    "/2025/Vectortile2026/river/{z}/{x}/{y}.pbf"
)

# The index is fully resolved at z=10; z>=14 returns 403.
INDEX_ZOOM = 10

# Bounding box of Saitama Prefecture (lat_south, lat_north, lon_west, lon_east).
SAITAMA_BBOX = (35.70, 36.30, 138.65, 140.00)

USER_AGENT = "AquaNexus/0.1 (research; +https://github.com/kiruthick01/AquaNexus)"


@dataclass(frozen=True)
class Tile:
    """One downloadable point-cloud tile (図郭)."""

    mesh: str
    url: str
    lon: float
    lat: float

    @property
    def river(self) -> str:
        """River group key, e.g. 'ayasegawa' from 'ayasegawa-0610'."""
        return self.mesh.rsplit("-", 2)[0]

    @property
    def sequence(self) -> int:
        """Trailing sequence number, used to order tiles along the river."""
        tail = self.mesh.rsplit("-", 1)[-1]
        return int(tail) if tail.isdigit() else -1


# ---------------------------------------------------------------------------
# Web-mercator tile arithmetic
# ---------------------------------------------------------------------------


def lonlat_to_tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    n = 2**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lonlat(x: int, y: int, z: int) -> tuple[float, float]:
    """North-west corner of tile (x, y) at zoom z."""
    n = 2**z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def _flatten_coords(coords) -> Iterator[tuple[float, float]]:
    """Yield (x, y) pairs from an arbitrarily nested GeoJSON coordinate list."""
    if coords and isinstance(coords[0], (int, float)):
        yield (coords[0], coords[1])
        return
    for part in coords:
        yield from _flatten_coords(part)


def _decode_tile(raw: bytes, x: int, y: int, z: int) -> Iterator[Tile]:
    import mapbox_vector_tile  # lazy: only needed when refreshing the index

    lon_w, lat_n = tile_to_lonlat(x, y, z)
    lon_e, lat_s = tile_to_lonlat(x + 1, y + 1, z)

    for layer in mapbox_vector_tile.decode(raw).values():
        extent = layer.get("extent", 4096)
        for feature in layer.get("features", []):
            props = feature.get("properties", {})
            mesh, url = props.get("MESH_NO"), props.get("URL")
            if not mesh or not url:
                continue
            pts = list(_flatten_coords(feature["geometry"]["coordinates"]))
            if not pts:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            yield Tile(
                mesh=mesh,
                url=url,
                lon=lon_w + (cx / extent) * (lon_e - lon_w),
                lat=lat_s + (cy / extent) * (lat_n - lat_s),
            )


def fetch_tile_index(
    bbox: tuple[float, float, float, float] = SAITAMA_BBOX,
    zoom: int = INDEX_ZOOM,
    pause: float = 0.05,
) -> list[Tile]:
    """Recover the tile index by decoding the published vector tiles.

    Roughly 15 requests for the whole prefecture. Prefer :func:`load_tile_index`,
    which caches the result.
    """
    lat_s, lat_n, lon_w, lon_e = bbox
    x0, y0 = lonlat_to_tile(lat_n, lon_w, zoom)
    x1, y1 = lonlat_to_tile(lat_s, lon_e, zoom)

    found: dict[str, Tile] = {}
    for x in range(x0, x1 + 1):
        for y in range(y0, y1 + 1):
            try:
                raw = _get(VECTOR_TILE_URL.format(z=zoom, x=x, y=y))
            except Exception as exc:  # a sparse grid legitimately has empty tiles
                log.debug("tile %d/%d/%d unavailable: %s", zoom, x, y, exc)
                continue
            for tile in _decode_tile(raw, x, y, zoom):
                found.setdefault(tile.mesh, tile)
            time.sleep(pause)

    log.info(
        "tile index: %d tiles across %d rivers",
        len(found),
        len({t.river for t in found.values()}),
    )
    return sorted(found.values(), key=lambda t: t.mesh)


def load_tile_index(refresh: bool = False) -> list[Tile]:
    """Return the tile index, fetching and caching it on first use."""
    cache = settings.CACHE_DIR / "saitama_tile_index.json"
    if cache.is_file() and not refresh:
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return [Tile(**r) for r in raw]

    tiles = fetch_tile_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps([asdict(t) for t in tiles], ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    return tiles


def tiles_for_river(river: str, tiles: Iterable[Tile] | None = None) -> list[Tile]:
    """Tiles belonging to one river, ordered along it by sequence number."""
    source = tiles if tiles is not None else load_tile_index()
    return sorted((t for t in source if t.river == river), key=lambda t: t.sequence)


# ---------------------------------------------------------------------------
# Download and load
# ---------------------------------------------------------------------------


def download_tile(tile: Tile, dest_dir: Path | None = None, overwrite: bool = False) -> Path:
    """Download one tile's zip. Skips work if the file is already present.

    Tiles are large - Ayase averages ~107 MB zipped, Naka ~616 MB.
    """
    dest_dir = dest_dir or (settings.RAW_DIR / "pointcloud" / tile.river)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{tile.mesh}.zip"

    if dest.is_file() and not overwrite:
        log.debug("%s already present, skipping", dest.name)
        return dest

    log.info("downloading %s", tile.mesh)

    # The scratch name must be unique per writer. A shared "<mesh>.part" lets two
    # processes downloading the same tile interleave: both write the one scratch
    # file and both rename it, so a partial download can land as a complete-looking
    # zip. That produced exactly one corrupt tile in an 89-tile run, and it only
    # surfaced later as a zlib "invalid distance code" during extraction.
    tmp = dest.with_suffix(f".{os.getpid()}.{threading.get_ident()}.part")
    try:
        req = urllib.request.Request(tile.url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)

        if not _is_valid_zip(tmp):
            raise OSError(f"{tile.mesh}: downloaded archive is corrupt")

        tmp.replace(dest)  # atomic: a partial download never looks complete
    finally:
        tmp.unlink(missing_ok=True)
    return dest


def _is_valid_zip(path: Path) -> bool:
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except Exception:  # noqa: BLE001 - any failure to read means unusable
        return False


def verify_tiles(river: str, dest_dir: Path | None = None,
                 repair: bool = False) -> list[Path]:
    """Check downloaded archives and optionally re-fetch the broken ones.

    Corruption is silent until extraction, by which point a long batch job has
    already been running for some time, so it is worth checking up front.
    Returns the paths that were bad.
    """
    dest_dir = dest_dir or (settings.RAW_DIR / "pointcloud" / river)
    if not dest_dir.is_dir():
        return []

    bad = [p for p in sorted(dest_dir.glob("*.zip")) if not _is_valid_zip(p)]
    log.info("%s: %d archive(s) checked, %d corrupt",
             river, len(list(dest_dir.glob("*.zip"))), len(bad))

    if repair and bad:
        index = {t.mesh: t for t in tiles_for_river(river)}
        for path in bad:
            tile = index.get(path.stem)
            if tile is None:
                log.error("no index entry for %s; cannot repair", path.name)
                continue
            log.info("re-downloading %s", tile.mesh)
            path.unlink(missing_ok=True)
            download_tile(tile, dest_dir, overwrite=True)
    return bad


def download_tiles(
    tiles: list[Tile],
    dest_dir: Path | None = None,
    workers: int = 6,
    overwrite: bool = False,
) -> list[Path]:
    """Download many tiles concurrently.

    Throughput on a single connection is limited by the server rather than the
    link - sequentially the full Ayase reach takes hours - so a small pool of
    workers cuts wall-clock time substantially. Kept modest to stay a polite
    client of a free public dataset.

    Failures are logged and skipped rather than aborting the batch; rerun to
    pick up what is missing, since completed files are skipped.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_tile, tile, dest_dir, overwrite): tile
            for tile in tiles
        }
        for done, future in enumerate(as_completed(futures), start=1):
            tile = futures[future]
            try:
                paths.append(future.result())
                log.info("[%d/%d] %s", done, len(tiles), tile.mesh)
            except Exception as exc:  # noqa: BLE001 - one bad tile must not stop the run
                log.error("[%d/%d] %s failed: %s", done, len(tiles), tile.mesh, exc)

    log.info("downloaded %d of %d tile(s)", len(paths), len(tiles))
    return paths


def extract_las(zip_path: Path, dest_dir: Path | None = None) -> Path:
    """Extract the single .las member from a tile zip."""
    import zipfile

    dest_dir = dest_dir or zip_path.parent
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".las")]
        if len(members) != 1:
            raise ValueError(
                f"expected exactly one .las in {zip_path.name}, found {members}"
            )
        member = members[0]
        expected = zf.getinfo(member).file_size
        out = dest_dir / Path(member).name

        # Existence alone is not proof of completeness: an interrupted extraction
        # leaves a truncated file that laspy rejects much later with
        # "buffer size must be a multiple of element size". Compare against the
        # size recorded in the archive and redo the work if it disagrees.
        if out.is_file() and out.stat().st_size == expected:
            return out
        if out.is_file():
            log.warning("%s is %d bytes, expected %d; re-extracting",
                        out.name, out.stat().st_size, expected)

        tmp = dest_dir / f"{out.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with zf.open(member) as src, tmp.open("wb") as fh:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            if tmp.stat().st_size != expected:
                raise OSError(f"{member}: extracted {tmp.stat().st_size} of {expected} bytes")
            tmp.replace(out)
        finally:
            tmp.unlink(missing_ok=True)
        return out


def load_points(las_path: Path, thin: int = 1) -> np.ndarray:
    """Load a LAS tile as an (N, 3) float64 array of easting, northing, elevation.

    Coordinates are EPSG:6677 (JGD2011 / Japan Plane Rectangular CS Zone 9) with
    x = easting and y = northing. This was verified against the tile index: the
    alternative (Japanese X=north) convention misplaces tiles by ~16 km.

    ``thin`` takes every n-th point, which is usually enough for cross-sections
    and keeps memory down - tiles carry roughly 27 points per square metre.
    """
    import laspy

    las = laspy.read(str(las_path))
    xyz = np.column_stack(
        (np.asarray(las.x), np.asarray(las.y), np.asarray(las.z))
    ).astype(np.float64)
    if thin > 1:
        xyz = xyz[::thin]
    log.debug("%s: %d points", las_path.name, len(xyz))
    return xyz
