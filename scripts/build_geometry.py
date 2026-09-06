"""Build a HEC-RAS model for a whole river reach from the Saitama point cloud.

    python scripts/build_geometry.py --river ayasegawa --spacing 500
    python scripts/build_geometry.py --river ayasegawa --limit 5 --spacing 100

Tiles are processed one at a time. Holding the full Ayase reach in memory at once
would be roughly 58 million points even after thinning, and only the sections
near a given tile can be cut from it anyway, so each tile is loaded, used and
released before the next is opened.

Downloads run concurrently: sequentially the reach fetches at about 16 MB/min
because throughput is capped per connection, against roughly 500 MB/min with a
small pool.

Data: 埼玉県 河川点群データ (CC BY 4.0).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from aquanexus.config import settings
from aquanexus.data.pointcloud import (
    download_tiles,
    extract_las,
    load_points,
    load_tile_index,
    tiles_for_river,
)
from aquanexus.data.validator import validate_sections
from aquanexus.hecras.geometry import (
    extract_cross_section,
    resample_centreline,
    smooth_centreline,
    to_hecras_geometry,
)
from aquanexus.hecras.project import SteadyFlowProfile, write_project
from aquanexus.logger import get_logger

log = get_logger("scripts.build_geometry")


def centreline_from_tiles(tiles) -> np.ndarray:
    """Project tile centroids to the project CRS to form a coarse channel axis.

    Centroids are tile centres rather than the thalweg, so the result is smoothed
    before sections are cut - jitter rotates the section normals, and a section
    cut at the wrong angle overstates channel width.
    """
    from pyproj import CRS, Transformer

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_epsg(settings.CRS_EPSG), always_xy=True
    )
    return np.asarray([transformer.transform(t.lon, t.lat) for t in tiles],
                      dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--river", default="ayasegawa")
    ap.add_argument("--spacing", type=float, default=500.0,
                    help="cross-section spacing in metres (default: 500)")
    ap.add_argument("--half-width", type=float, default=80.0)
    ap.add_argument("--bed-quantile", type=float, default=0.05)
    ap.add_argument("--thin", type=int, default=4, help="use every n-th point")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N tiles - for a trial run")
    ap.add_argument("--workers", type=int, default=6, help="concurrent downloads")
    ap.add_argument("--smooth", type=int, default=5, help="centreline smoothing window")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--no-run", action="store_true",
                    help="write the project but do not compute")
    args = ap.parse_args()

    settings.ensure_dirs()

    tiles = tiles_for_river(args.river, load_tile_index())
    if not tiles:
        log.error("no tiles found for river %r", args.river)
        return 1
    if args.limit:
        tiles = tiles[: args.limit]
    log.info("%s: %d tile(s)", args.river, len(tiles))

    download_tiles(tiles, workers=args.workers)

    # Section stations are laid out once, along the whole reach.
    line = smooth_centreline(centreline_from_tiles(tiles), args.smooth)
    origins, tangents = resample_centreline(line, args.spacing)
    log.info("centreline %.1f km -> %d candidate sections",
             np.hypot(*np.diff(line, axis=0).T).sum() / 1000.0, len(origins))

    # One tile at a time: cut only the sections that fall inside it.
    sections = []
    margin = args.half_width
    for index, tile in enumerate(tiles, start=1):
        try:
            points = load_points(extract_las(tile_zip(tile)), thin=args.thin)
        except FileNotFoundError:
            log.warning("[%d/%d] %s not downloaded, skipping", index, len(tiles), tile.mesh)
            continue

        x0, x1 = points[:, 0].min() - margin, points[:, 0].max() + margin
        y0, y1 = points[:, 1].min() - margin, points[:, 1].max() + margin
        inside = np.where(
            (origins[:, 0] >= x0) & (origins[:, 0] <= x1)
            & (origins[:, 1] >= y0) & (origins[:, 1] <= y1)
        )[0]

        found = 0
        for i in inside:
            xs = extract_cross_section(
                points, origins[i], tangents[i],
                river_station=float(i * args.spacing),
                half_width=args.half_width,
                bed_quantile=args.bed_quantile,
            )
            if not xs.is_empty:
                sections.append(xs)
                found += 1
        log.info("[%d/%d] %s: %d section(s)", index, len(tiles), tile.mesh, found)
        del points

    if not sections:
        log.error("no cross-sections extracted")
        return 1

    # Keep the deepest sample per river station: tiles overlap at their edges.
    best: dict[float, object] = {}
    for xs in sections:
        current = best.get(xs.river_station)
        if current is None or xs.n_points > current.n_points:
            best[xs.river_station] = xs
    sections = sorted(best.values(), key=lambda s: s.river_station)
    log.info("%d unique section(s) after de-duplication", len(sections))

    report = validate_sections(sections)
    print(report)

    out_dir = args.out or (settings.HECRAS_DIR / args.river)
    profiles = [SteadyFlowProfile("Low", 0.25),
                SteadyFlowProfile("Median", 9.7),
                SteadyFlowProfile("High", 64.2)]
    prj = write_project(
        out_dir, args.river.capitalize(),
        to_hecras_geometry(sections, river=args.river.capitalize(), reach="Main"),
        profiles, river=args.river.capitalize(), reach="Main",
        upstream_station=max(s.river_station for s in sections),
    )
    log.info("wrote %s", prj)

    if args.no_run:
        return 0

    from aquanexus.hecras.runner import RasController

    with RasController(prj) as ras:
        ok, messages = ras.compute()
        if not ok:
            log.error("compute failed; see %s.computeMsgs.txt", prj.stem)
            for m in messages:
                log.error("  %s", m)
            return 1
        for i, profile in enumerate(profiles, start=1):
            rows = ras.profile_results(profile=i)
            depths = [r["depth"] for r in rows if np.isfinite(r["depth"])]
            log.info("%s (Q=%s): %d section(s), depth %.2f-%.2f m",
                     profile.name, profile.discharge, len(rows),
                     min(depths), max(depths))
    return 0


def tile_zip(tile):
    """Where :func:`download_tile` puts a tile."""
    return settings.RAW_DIR / "pointcloud" / tile.river / f"{tile.mesh}.zip"


if __name__ == "__main__":
    sys.exit(main())
