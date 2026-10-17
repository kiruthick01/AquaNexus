"""Fetch GSI elevation at every MOE station (resumable).

    python scripts/fetch_elevation.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aquanexus.data import geography, moe
from aquanexus.logger import get_logger

log = get_logger("scripts.fetch_elevation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/moe"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/elevation.csv"))
    args = parser.parse_args()

    samples = moe.load_samples(moe.find_files(args.raw, "mk"))
    stations = moe.load_stations(moe.find_files(args.raw, "mm"))
    data = moe.build_dataset(samples, stations)
    used = stations[stations["station_id"].isin(data["station_id"].unique())]
    log.info("%d stations", len(used))

    out = geography.fetch_elevation(used, args.cache)
    log.info("%d stations cached in %s", len(out), args.cache)


if __name__ == "__main__":
    main()
