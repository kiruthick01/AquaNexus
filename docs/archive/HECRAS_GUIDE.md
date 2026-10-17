# HEC-RAS Guide

How AquaNexus builds and runs HEC-RAS models, and the file-format details that
are not in the documentation. Everything here was verified against **HEC-RAS 7.0
(April 2026)** on Windows.

---

## Setup

1. Install HEC-RAS 6.x or 7.x from [hec.usace.army.mil](https://www.hec.usace.army.mil/software/hec-ras/).
   Windows only — there is no Linux build and no headless mode.
2. Install the automation extra: `pip install -e ".[hecras]"` (pulls in `pywin32`).
3. Point `HECRAS_EXE` at your install, or let `find_hecras_exe()` locate it:

```python
from aquanexus.hecras.runner import find_hecras_exe
find_hecras_exe()   # scans the install root, newest version first
```

The COM server registers as `RAS70.HECRASController`; `RasController` tries
several ProgIDs so a version bump does not break the pipeline.

---

## Running the pipeline

```python
from aquanexus.data.pointcloud import load_tile_index, tiles_for_river
from aquanexus.hecras.geometry import extract_sections, to_hecras_geometry
from aquanexus.hecras.project import write_project, SteadyFlowProfile
from aquanexus.hecras.runner import RasController

sections = extract_sections(points, centreline, spacing=500.0)

prj = write_project(
    "data/hecras/ayase", "Ayase",
    to_hecras_geometry(sections, river="Ayase", reach="Main"),
    profiles=[SteadyFlowProfile("Low", 0.25),
              SteadyFlowProfile("Median", 9.7),
              SteadyFlowProfile("High", 64.2)],
    river="Ayase", reach="Main",
    upstream_station=max(s.river_station for s in sections),
)

with RasController(prj) as ras:
    ok, messages = ras.compute()
    rows = ras.profile_results(profile=2)   # depth, velocity, area, top width
```

Or from the command line: `python scripts/build_geometry.py --river ayasegawa --spacing 500`

---

## File format notes

These cost real time to find. Every one produced a silent failure rather than a
clear error.

### Geometry (`.g01`)

| Requirement | Symptom if wrong |
|---|---|
| River **and** reach names padded to exactly 16 characters | Loads with **zero rivers**, no error |
| `Reach XY` centreline block present | Same |
| Per section: `Node Last Edited Time`, `#Mann`, `Bank Sta`, `XS Rating Curve`, `XS HTab Starting El and Incr`, `XS HTab Horizontal Distribution`, `Exp/Cntr` | Section treated as incomplete |
| File trailer (`LCMann …`, `Chan Stop Cuts`, `Composite Channel Slope`, …) | Same |
| Station/elevation as 8-character fields, five pairs per line | Parse corruption |
| **Bank stations must be values present in the station/elevation list** | `Left bank station not in station elevation data`, run refused |
| Downstream-most section takes reach lengths `0,0,0`; all others positive | `Main channel length is less than or equal to zero` |
| Sections written in **descending** river station | Ordering errors |

The bank-station rule is the one that bites. An interpolated "35% of the way
across" value is not in the list — and extraction drops sparse bins, so the list
has gaps. `bank_stations()` walks outward from the thalweg instead, which returns
real stations by construction.

### Flow (`.f01`)

The `.f01` and `.g01` **disagree on padding**:

```
.g01   River Reach=Ayase           ,Main
.f01   River Rch & RM=Ayase,Main            ,300
```

Geometry pads both names to 16. The flow file pads only the reach and leaves the
river name bare. Matching the geometry here is wrong.

Flow is specified only at the upstream-most station; HEC-RAS routes it downstream.
Each profile needs its own `Boundary for River Rch & Prof#` block.

### Plan (`.p01`)

**A hand-written plan will not run.** HEC-RAS 7.0's own plan carries around 200
settings, and a minimal one is rejected with:

```
Computations were not performed, there must have been some
missing data in the input files.
```

which names neither the file nor the field. `src/aquanexus/hecras/templates/`
holds a plan captured from a reference project HEC-RAS wrote and computed itself;
only the title, short identifier and file extensions are substituted.

### Project (`.prj`)

- `SI Units` is required for metric input. Without it HEC-RAS assumes feet and the
  numbers are quietly wrong rather than rejected.
- `Current Plan=p01` is required. **The plan selects the geometry, not the
  project** — a `.prj` naming `Geom File=g01` still yields an empty
  `CurrentGeomFile` and zero rivers without it.
- Never reference a file that does not exist. `Unsteady File=u01` with no `.u01`
  present raises a **modal dialog**; over COM that blocks the caller indefinitely
  with no error and leaves orphaned `Ras.exe` processes.

`validate_project()` checks all three before anything opens.

### Line endings

HEC-RAS files are CRLF. Writing an already-CRLF string with Python's default text
mode emits `\r\r\n`, because the text layer translates the `\n` a second time.
HEC-RAS **accepts** that file and then hangs on a dialog. Always pass
`newline=""`.

---

## Debugging a failed run

The COM controller returns only the generic summary. **The real diagnosis is
written to `<project>.<plan>.computeMsgs.txt`** next to the project:

```
Message  | River: "Ayase"   Reach: "Main"   RS:250
Message  |      - Left bank station not in station elevation data.
```

Read that file first. Four attempts were wasted guessing before it was found.

The GUI shows the same detail in an error window, so opening the project and
pressing Compute is a good fallback when automation is opaque.

---

## Reading results

`Output_NodeOutput` returns `(value, *echoed arguments)`, not a scalar, and
signals "no value" with a sentinel near the maximum single-precision float rather
than raising. Variable ids are undocumented; those in `runner.Output` were
identified by matching returned values against known quantities — velocity, for
instance, is the id whose value equals discharge divided by flow area.

| Id | Quantity |
|---|---|
| 2 | Water surface elevation (m) |
| 3 | Energy grade elevation (m) |
| 4 | Maximum channel depth (m) |
| 5 | Minimum channel elevation / invert (m) |
| 7 | Total discharge (m³/s) |
| 10 | Flow area (m²) |
| 14 | Top width (m) |
| 23 | Channel velocity (m/s) |
| 30 | Energy (friction) slope |

---

## Limitations

- **Steady flow only.** Unsteady and the water-quality modules (NSMI/NSMII) are
  not driven by this package. NSMI needs per-constituent boundary time series that
  the monthly monitoring record cannot supply — see `docs/DATA_SOURCES.md`.
- **Manning's n is assumed**, not calibrated: 0.035 in channel, 0.06 overbank.
  With no gauged rating curve for the modelled reach there is nothing to calibrate
  against, so absolute depths carry that uncertainty. Relative behaviour across
  the flow sweep is more trustworthy than any single depth.
- **Cross-sections come from a point cloud with no ground classification**, so the
  bed is a low elevation quantile per bin rather than surveyed. Run
  `validate_sections()` and inspect anything it flags before trusting a reach.
