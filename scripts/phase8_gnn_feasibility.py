"""Phase 8 GNN feasibility check - run against the real station lists.

Does not train any GCN/GAT on real data. Loads the real Ayase and Naka
canonical datasets, collects their actual station names, and runs
``aquanexus.ml.graph.require_station_topology`` against each river - the
pipeline's entry guard. Also confirms directly, rather than by inspection
alone, that the one topology-like field this repository does carry
(`river_station` from the HEC-RAS sweep) cannot supply a monitoring-station
edge list: it orders cross-sections along a continuous reach, and no join
from a monitoring station's name to a specific `river_station` value exists
anywhere in `aquanexus.data.dataset`.

    python scripts/phase8_gnn_feasibility.py
"""

from __future__ import annotations

import glob
import sys

import pandas as pd

from aquanexus.config import settings
from aquanexus.data.dataset import build_water_quality_dataset
from aquanexus.data.loader import filter_stations, load_many
from aquanexus.logger import get_logger
from aquanexus.ml.graph import MissingTopologyError, require_station_topology

log = get_logger("scripts.phase8_gnn_feasibility")


def load_stations_and_sweep(water_body: str, sweep_name: str) -> tuple[list[str], pd.DataFrame]:
    files = sorted(glob.glob(str(settings.RAW_DIR / "waterquality" / "saitama_*.xlsx")))
    if not files:
        raise FileNotFoundError("no water quality files; run scripts/download_data.py")
    sweep_path = settings.PROCESSED_DIR / f"{sweep_name}.csv"
    if not sweep_path.is_file():
        raise FileNotFoundError(f"no sweep at {sweep_path}")
    observations = filter_stations(load_many(files), water_body=water_body)
    sweep = pd.read_csv(sweep_path)
    dataset = build_water_quality_dataset(observations, sweep)
    return sorted(dataset["station"].unique().tolist()), sweep


def main() -> int:
    ayase_stations, ayase_sweep = load_stations_and_sweep(settings.RIVER_NAME_JA,
                                                          "ayase_flow_sweep")
    naka_stations, naka_sweep = load_stations_and_sweep("中川", "naka_flow_sweep")

    log.info("Ayase: %d station(s), %d distinct river_station value(s) in the sweep "
             "(cross-sections, not monitoring stations).",
             len(ayase_stations), ayase_sweep["river_station"].nunique())
    log.info("Naka: %d station(s), %d distinct river_station value(s) in the sweep.",
             len(naka_stations), naka_sweep["river_station"].nunique())
    log.info("Neither sweep's river_station column is joined to a monitoring station "
             "name anywhere in aquanexus.data.dataset - confirmed by inspection of "
             "build_water_quality_dataset, which uses river_station only to average "
             "hydraulics across a reach, never to identify a specific station's position.")

    print()
    print("=" * 70)
    blocked_any = False
    for river, stations in (("Ayase", ayase_stations), ("Naka", naka_stations)):
        try:
            require_station_topology(stations, river=river)
        except MissingTopologyError as exc:
            blocked_any = True
            log.error("PHASE 8 GNN: BLOCKED for %s\n%s", river, str(exc))
    if blocked_any:
        print("PHASE 8 GNN: BLOCKED BY MISSING TOPOLOGY (both rivers)")
        print(f"Ayase: {len(ayase_stations)} candidate node(s), no edge list.")
        print(f"Naka: {len(naka_stations)} candidate node(s), no edge list.")
        print("No GCN/GAT was trained on real station data. See docs/DATA_LIMITATIONS.md "
              "Phase 8 and the logged errors above for detail.")
    else:
        print("UNEXPECTED: topology is now on record for at least one river.")
        print("Re-check docs/ML_ROADMAP.md Phase 8 status before proceeding.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
