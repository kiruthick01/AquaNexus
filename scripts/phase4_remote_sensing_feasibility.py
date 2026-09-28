"""Phase 4 remote-sensing feasibility check - run against the real station list.

Does not fetch any satellite imagery. Loads the real Ayase and Naka canonical
datasets, collects their actual station names, and runs
``aquanexus.remote_sensing.require_station_coordinates`` against them - the
pipeline's entry guard. This is the concrete evidence behind Phase 4's
"blocked by missing station coordinates" status: not every function in
``remote_sensing.py`` is untested (see ``tests/test_remote_sensing.py``), only
that none of them can be pointed at a real AquaNexus station yet.

    python scripts/phase4_remote_sensing_feasibility.py
"""

from __future__ import annotations

import glob
import sys

import pandas as pd

from aquanexus.config import settings
from aquanexus.data.dataset import build_water_quality_dataset
from aquanexus.data.loader import filter_stations, load_many
from aquanexus.logger import get_logger
from aquanexus.remote_sensing import MissingStationCoordinatesError, require_station_coordinates

log = get_logger("scripts.phase4_remote_sensing_feasibility")


def load_stations(water_body: str, sweep_name: str) -> list[str]:
    files = sorted(glob.glob(str(settings.RAW_DIR / "waterquality" / "saitama_*.xlsx")))
    if not files:
        raise FileNotFoundError("no water quality files; run scripts/download_data.py")
    sweep_path = settings.PROCESSED_DIR / f"{sweep_name}.csv"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"no sweep at {sweep_path}")
    observations = filter_stations(load_many(files), water_body=water_body)
    dataset = build_water_quality_dataset(observations, pd.read_csv(sweep_path))
    return sorted(dataset["station"].unique().tolist())


def main() -> int:
    ayase_stations = load_stations(settings.RIVER_NAME_JA, "ayase_flow_sweep")
    naka_stations = load_stations("中川", "naka_flow_sweep")
    all_stations = sorted(set(ayase_stations) | set(naka_stations))
    log.info("Ayase: %d station(s). Naka: %d station(s). Combined: %d unique.",
             len(ayase_stations), len(naka_stations), len(all_stations))

    print()
    print("=" * 70)
    try:
        require_station_coordinates(all_stations)
    except MissingStationCoordinatesError as exc:
        # Station names are Japanese; route through the logger (UTF-8
        # reconfigured stream, see aquanexus.logger), not print(), which
        # inherits the console's cp1252 default on Windows.
        log.error("PHASE 4 REMOTE SENSING: BLOCKED BY MISSING STATION COORDINATES\n%s",
                 str(exc))
        print("PHASE 4 REMOTE SENSING: BLOCKED BY MISSING STATION COORDINATES")
        print("No satellite imagery was fetched. No NDVI/NDWI/MNDWI feature was")
        print("computed for any real station. See docs/DATA_LIMITATIONS.md Phase 4")
        print("and the logged error above for exactly which stations are missing.")
        print("=" * 70)
        return 0
    print("UNEXPECTED: station coordinates are now on record.")
    print("Re-check docs/ML_ROADMAP.md Phase 4 status before proceeding to feature extraction.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
