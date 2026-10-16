"""Fetch NASA POWER daily weather for every grid cell that holds a MOE station.

Resumable: cells already in ``data/cache/weather`` are skipped, so rerun after an
interruption or a rate-limit stop. Fetching starts 30 days before the first sample
so the antecedent-rainfall windows are complete from day one.

    python scripts/fetch_weather.py
"""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

from aquanexus.data import moe, weather
from aquanexus.logger import get_logger

log = get_logger("scripts.fetch_weather")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=Path("data/raw/moe"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/weather"))
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests")
    parser.add_argument("--limit", type=int, default=None, help="fetch at most N new cells")
    args = parser.parse_args()

    samples = moe.load_samples(moe.find_files(args.raw, "mk"))
    stations = moe.load_stations(moe.find_files(args.raw, "mm"))
    data = moe.build_dataset(samples, stations)

    start = (data["date"].min() - timedelta(days=30)).strftime("%Y-%m-%d")
    end = data["date"].max().strftime("%Y-%m-%d")
    cells = weather.station_cells(data)[["cell_lat", "cell_lon"]].drop_duplicates()
    todo = list(cells.itertuples(index=False, name=None))
    if args.limit:
        todo = [c for c in todo if not (args.cache / f"{weather.cell_key(*c)}.csv").exists()]
        todo = todo[: args.limit]
    log.info("%d cells, %s to %s, cache %s", len(todo), start, end, args.cache)

    fetched = weather.fetch_cells(todo, start, end, args.cache, pause=args.pause)
    have = len(list(args.cache.glob("*.csv")))
    log.info("fetched %d this run; %d cells cached of %d needed", fetched, have, len(cells))


if __name__ == "__main__":
    main()
