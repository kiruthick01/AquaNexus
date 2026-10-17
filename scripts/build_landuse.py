"""Compute land use shares around every station from the downloaded mesh files.

    python scripts/build_landuse.py

Reads ``data/raw/landuse/`` (see ``scripts/fetch_landuse.py``) and writes
``data/cache/landuse.csv``: per station, the share of each land use class within 1 km and
5 km, computed over land cells. A station whose mesh file is missing is left out and
simply has no land use features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aquanexus.data import landuse, moe
from aquanexus.logger import get_logger

log = get_logger("scripts.build_landuse")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/moe"))
    parser.add_argument("--tiles", type=Path, default=Path("data/raw/landuse"))
    parser.add_argument("--out", type=Path, default=Path("data/cache/landuse.csv"))
    args = parser.parse_args()

    samples = moe.load_samples(moe.find_files(args.raw, "mk"))
    stations = moe.load_stations(moe.find_files(args.raw, "mm"))
    used = moe.build_dataset(samples, stations).drop_duplicates("station_id")
    table = landuse.station_landuse(used[["station_id", "latitude", "longitude"]], args.tiles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    log.info("%d of %d stations have land use features", len(table), len(used))


if __name__ == "__main__":
    main()
