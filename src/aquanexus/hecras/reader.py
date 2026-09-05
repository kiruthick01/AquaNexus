"""Read HEC-RAS project and geometry files.

HEC-RAS stores a project as a set of plain-text sidecar files sharing one stem:
``.prj`` (project), ``.g01``.. (geometry), ``.f01``/``.u01`` (steady/unsteady
flow), ``.p01`` (plan). Results land in HDF5 alongside them.

This module covers the text files needed to inspect a project and to read back
geometry - including geometry this package wrote, which makes the export in
:mod:`aquanexus.hecras.geometry` round-trippable and therefore testable.

Result HDF5 parsing belongs in ``parser.py`` and is not implemented yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aquanexus.hecras.geometry import CrossSection
from aquanexus.logger import get_logger

log = get_logger("hecras.reader")

# HEC-RAS writes station/elevation as fixed-width 8-character fields.
_FIELD_WIDTH = 8


@dataclass
class RasProject:
    """Metadata parsed from a ``.prj`` file."""

    path: Path
    title: str = ""
    description: str = ""
    geometry_files: list[str] = field(default_factory=list)
    flow_files: list[str] = field(default_factory=list)
    plan_files: list[str] = field(default_factory=list)

    @property
    def stem_dir(self) -> Path:
        return self.path.parent

    def geometry_paths(self) -> list[Path]:
        return [self.stem_dir / g for g in self.geometry_files]


@dataclass
class RasReach:
    """One river/reach and its cross-sections."""

    river: str
    reach: str
    sections: list[CrossSection] = field(default_factory=list)

    @property
    def length(self) -> float:
        """Distance between the extreme river stations."""
        if len(self.sections) < 2:
            return 0.0
        rs = [s.river_station for s in self.sections]
        return max(rs) - min(rs)


def _split_fixed_width(line: str, width: int = _FIELD_WIDTH) -> list[float]:
    """Split a fixed-width numeric line, tolerating a ragged final field."""
    out = []
    for i in range(0, len(line), width):
        chunk = line[i : i + width].strip()
        if not chunk:
            continue
        try:
            out.append(float(chunk))
        except ValueError:
            log.debug("skipping non-numeric field %r", chunk)
    return out


def read_project(path: Path | str) -> RasProject:
    """Parse a HEC-RAS ``.prj`` file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    project = RasProject(path=path)
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()

        if key == "Proj Title":
            project.title = value
        elif key == "Description":
            project.description = value
        elif key.startswith("Geom File"):
            project.geometry_files.append(value)
        elif key.startswith(("Flow File", "Unsteady File")):
            project.flow_files.append(value)
        elif key.startswith("Plan File"):
            project.plan_files.append(value)

    log.info(
        "%s: %d geometry, %d flow, %d plan file(s)",
        path.name,
        len(project.geometry_files),
        len(project.flow_files),
        len(project.plan_files),
    )
    return project


def read_geometry(path: Path | str) -> list[RasReach]:
    """Parse a HEC-RAS ``.g01`` geometry file into reaches of cross-sections.

    Only the station/elevation profile of each cross-section is recovered -
    enough to inspect and validate channel shape. Bank stations, ineffective
    areas, levees and bridge data are skipped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    reaches: list[RasReach] = []
    current: RasReach | None = None
    river_station = 0.0
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("River Reach="):
            value = line.split("=", 1)[1]
            river, _, reach = value.partition(",")
            current = RasReach(river=river.strip(), reach=reach.strip())
            reaches.append(current)

        elif line.startswith("Type RM Length"):
            # "Type RM Length L Ch R = 1 ,<river station>,,,"
            parts = line.split("=", 1)[1].split(",")
            river_station = _first_float(parts[1:]) if len(parts) > 1 else 0.0

        elif line.startswith("#Sta/Elev="):
            count = int(line.split("=", 1)[1].strip())
            values: list[float] = []
            i += 1
            # Each pair is two fields, so collect until we have 2 * count values.
            while i < len(lines) and len(values) < count * 2:
                body = lines[i]
                if body.strip().startswith("#") or "=" in body:
                    break
                values.extend(_split_fixed_width(body))
                i += 1
            i -= 1

            pairs = np.asarray(values[: count * 2]).reshape(-1, 2)
            if current is None:  # geometry without a River Reach header
                current = RasReach(river="Unknown", reach="Unknown")
                reaches.append(current)
            current.sections.append(
                CrossSection(
                    river_station=river_station,
                    station=pairs[:, 0],
                    elevation=pairs[:, 1],
                    origin=(0.0, 0.0),
                    direction=(1.0, 0.0),
                    n_points=len(pairs),
                )
            )

        i += 1

    total = sum(len(r.sections) for r in reaches)
    log.info("%s: %d reach(es), %d cross-section(s)", path.name, len(reaches), total)
    return reaches


def _first_float(parts: list[str]) -> float:
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            return float(p)
        except ValueError:
            continue
    return 0.0


def find_projects(root: Path | str) -> list[Path]:
    """Find HEC-RAS ``.prj`` files under a directory.

    ``.prj`` is also an ESRI projection extension, so candidates are checked for
    a HEC-RAS marker rather than trusted on extension alone.
    """
    root = Path(root)
    found = []
    for candidate in sorted(root.rglob("*.prj")):
        try:
            head = candidate.read_text(encoding="utf-8", errors="replace")[:2048]
        except OSError:
            continue
        if re.search(r"^\s*(Proj Title|Geom File)\s*=", head, re.MULTILINE):
            found.append(candidate)
    return found
