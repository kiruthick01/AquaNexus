"""Run the flow sweep and write the table the models are built from.

`build_geometry.py` cuts cross-sections from the point cloud and computes three
profiles as a sanity check. This is the other half: it takes an existing
geometry and runs the twelve log-spaced discharges spanning the observed range,
producing `data/processed/ayase_flow_sweep.csv` - the file
`aquanexus.data.dataset` interpolates for every training row and the API reads
to carry hydraulics through a scenario.

Log spacing matters. Discharge on this river covers two and a half orders of
magnitude, so linear spacing would put almost every profile at the high end and
misrepresent low flow, which is where habitat stress bites.

    python scripts/run_hecras.py                    # all sections
    python scripts/run_hecras.py --exclude-flagged  # drop validator warnings
    python scripts/run_hecras.py --no-run           # rebuild the CSV only

`--exclude-flagged` drops the sections the validator calls constrictions - cuts
that caught a slice of bank where the centreline wandered off the channel.
`docs/CONSTRICTION_IMPACT.md` measures what that is worth: 1.10 mg/L at worst,
64% of the model's error, which is why it is the default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from aquanexus.config import settings
from aquanexus.data.validator import validate_sections
from aquanexus.hecras.geometry import to_hecras_geometry
from aquanexus.hecras.project import SteadyFlowProfile, write_project
from aquanexus.hecras.reader import read_geometry
from aquanexus.logger import get_logger

log = get_logger("scripts.run_hecras")

#: Twelve log-spaced discharges spanning the observed range (0.17-73.7 m3/s).
DISCHARGES = (0.17, 0.3, 0.51, 0.89, 1.55, 2.69, 4.67, 8.1, 14.07, 24.44,
              42.44, 73.71)

COLUMNS = ["discharge_bc", "river_station", "wse", "invert", "depth", "velocity",
           "discharge", "flow_area", "top_width", "energy_slope"]


def constriction_stations(sections) -> list[float]:
    """River stations the validator flags as cut through a constriction."""
    report = validate_sections(sections)
    stations = []
    for issue in report.warnings:
        if issue.check != "section.constriction":
            continue
        try:
            stations.append(float(issue.location.removeprefix("RS ").strip()))
        except ValueError:
            log.warning("unreadable station in %r", issue.location)
    return sorted(set(stations))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default=None,
                        help="source .g01 (default: the shipped sweep project)")
    parser.add_argument("--out", default=None, help="HEC-RAS working directory")
    parser.add_argument("--exclude-flagged", action="store_true",
                        help="drop sections the validator flags as constrictions")
    parser.add_argument("--no-run", action="store_true",
                        help="rebuild the CSV from a previous computation")
    args = parser.parse_args()

    settings.ensure_dirs()
    geometry = Path(args.geometry) if args.geometry else (
        settings.HECRAS_DIR / "ayase_sweep" / "Ayase.g01")
    if not geometry.is_file():
        log.error("no geometry at %s; run scripts/build_geometry.py first", geometry)
        return 1

    sections = sorted(read_geometry(geometry)[0].sections,
                      key=lambda s: s.river_station)
    log.info("%d section(s) from %s", len(sections), geometry.name)

    if args.exclude_flagged:
        dropped = constriction_stations(sections)
        sections = [s for s in sections if s.river_station not in dropped]
        log.info("dropped %d flagged section(s): %s -> %d remain",
                 len(dropped), ", ".join(f"RS {d:.0f}" for d in dropped),
                 len(sections))

    out_dir = Path(args.out) if args.out else (settings.HECRAS_DIR / "ayase_sweep")
    profiles = [SteadyFlowProfile("Q" + f"{q}".replace(".", "p"), q)
                for q in DISCHARGES]

    prj = write_project(
        out_dir, "Ayase",
        to_hecras_geometry(sections, river="Ayasegawa", reach="Main"),
        profiles, river="Ayasegawa", reach="Main",
        upstream_station=max(s.river_station for s in sections),
    )
    log.info("wrote %s", prj)

    if args.no_run:
        log.info("--no-run: not computing")
        return 0

    from aquanexus.hecras.runner import RasController

    rows = []
    with RasController(prj) as ras:
        ok, messages = ras.compute()
        if not ok:
            log.error("compute failed")
            for message in messages[:10]:
                log.error("  %s", message)
            return 1
        for index, profile in enumerate(profiles, start=1):
            results = ras.profile_results(profile=index)
            rows.extend({**row, "discharge_bc": profile.discharge} for row in results)
            log.info("%s (Q=%s): %d section(s)", profile.name, profile.discharge,
                     len(results))

    sweep = pd.DataFrame(rows)[COLUMNS]
    out_csv = settings.PROCESSED_DIR / "ayase_flow_sweep.csv"
    sweep.to_csv(out_csv, index=False)
    log.info("wrote %s: %d row(s), %d section(s) x %d discharge(s)",
             out_csv, len(sweep), sweep.river_station.nunique(),
             sweep.discharge_bc.nunique())

    log.info("retrain to pick this up: python scripts/train_models.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
