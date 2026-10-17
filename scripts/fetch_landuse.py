"""Download the land use mesh files that cover the MOE stations (resumable).

    python scripts/fetch_landuse.py [--dry-run]

Fetches 国土数値情報 L03-b (FY2016, 100 m mesh, world datum) GML zips, one per 1st-order
mesh containing at least one station, into ``data/raw/landuse/``. Requests are
sequential with a pause. ``--dry-run`` lists what would be fetched and the total size.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aquanexus.data import landuse, moe
from aquanexus.logger import get_logger

log = get_logger("scripts.fetch_landuse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/moe"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/landuse"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    samples = moe.load_samples(moe.find_files(args.raw, "mk"))
    stations = moe.load_stations(moe.find_files(args.raw, "mm"))
    data = moe.build_dataset(samples, stations)
    used = data.drop_duplicates("station_id")
    codes = set(landuse.mesh1_code(used["latitude"], used["longitude"]).tolist())

    files = landuse.select_files(landuse.fetch_listing(), codes)
    missing = sorted(codes - set(files["code"]))
    log.info("%d meshes needed, %d files, %.0f MB; no file for %s", len(codes), len(files),
             files["mb"].sum(), missing)  # fmt: skip
    if not args.dry_run:
        landuse.download(files, args.out)


if __name__ == "__main__":
    main()
