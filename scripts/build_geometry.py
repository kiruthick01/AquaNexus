"""Build a HEC-RAS geometry file from the Saitama river point cloud.

Downloads the tiles for one river, derives a centreline from the tile index,
cuts cross-sections at a fixed spacing and writes a .g01.

Be aware of the download volume before running without --limit:
Ayase is ~9.5 GB across 89 tiles; Naka is ~32.7 GB across 53.

    python scripts/build_geometry.py --river ayasegawa --limit 5 --spacing 200
    python scripts/build_geometry.py --river ayasegawa --spacing 500

Data: 埼玉県 河川点群データ (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from aquanexus.config import settings
from aquanexus.data.pointcloud import (
    download_tile,
    extract_las,
    load_points,
    load_tile_index,
    tiles_for_river,
)
from aquanexus.hecras.geometry import extract_sections, to_hecras_geometry
from aquanexus.logger import get_logger

log = get_logger("scripts.build_geometry")


def centreline_from_tiles(tiles) -> np.ndarray:
    """Project tile centroids to EPSG:6677 to form a coarse channel axis.

    Centroids only approximate the channel - they are tile centres, not the
    thalweg - so the result is smoothed before sections are cut.
    """
    from pyproj import CRS, Transformer

    tr = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(settings.CRS_EPSG),
                             always_xy=True)
    xy = [tr.transform(t.lon, t.lat) for t in tiles]
    return np.asarray(xy, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--river", default="ayasegawa", help="river key, e.g. ayasegawa")
    ap.add_argument("--spacing", type=float, default=500.0,
                    help="cross-section spacing in metres (default: 500)")
    ap.add_argument("--half-width", type=float, default=80.0,
                    help="half-width of each section in metres (default: 80)")
    ap.add_argument("--bed-quantile", type=float, default=0.05,
                    help="elevation quantile taken as bed within each bin")
    ap.add_argument("--thin", type=int, default=4,
                    help="use every n-th point (default: 4)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N tiles - use for a trial run")
    ap.add_argument("--out", default=None, help="output .g01 path")
    args = ap.parse_args()

    settings.ensure_dirs()

    tiles = tiles_for_river(args.river, load_tile_index())
    if not tiles:
        log.error("no tiles found for river %r", args.river)
        return 1
    if args.limit:
        tiles = tiles[: args.limit]

    log.info("%s: %d tile(s)", args.river, len(tiles))

    clouds = []
    for i, tile in enumerate(tiles, 1):
        log.info("[%d/%d] %s", i, len(tiles), tile.mesh)
        las = extract_las(download_tile(tile))
        clouds.append(load_points(las, thin=args.thin))

    points = np.vstack(clouds)
    log.info("combined cloud: %d points", len(points))

    sections = extract_sections(
        points,
        centreline_from_tiles(tiles),
        spacing=args.spacing,
        half_width=args.half_width,
        bed_quantile=args.bed_quantile,
    )
    if not sections:
        log.error("no cross-sections extracted - check spacing and half-width")
        return 1

    out = args.out or (settings.HECRAS_DIR / f"{args.river}.g01")
    text = to_hecras_geometry(sections, river=args.river.capitalize(), reach="Main")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)

    inverts = [s.thalweg for s in sections]
    log.info("wrote %s", out)
    log.info(
        "%d sections | invert %.2f..%.2f m | mean sampled width %.0f m",
        len(sections), min(inverts), max(inverts),
        sum(s.width for s in sections) / len(sections),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
