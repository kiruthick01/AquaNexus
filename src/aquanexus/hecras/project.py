"""Write a complete, runnable HEC-RAS project.

HEC-RAS will not compute from a hand-minimised project. Its plan file carries
around 200 settings and omitting them yields only::

    Computations were not performed, there must have been some
    missing data in the input files.

- which names neither the file nor the field. Rather than reverse-engineer which
of those keys are load-bearing, the plan and project files here are templates
captured from a reference project that HEC-RAS 7.0 itself wrote and successfully
computed; only title, short identifier and the geometry/flow extensions are
substituted. The flow file is generated, because its content genuinely varies.

Formatting rules, taken from that reference and not guessable
-------------------------------------------------------------
* ``.g01`` pads **both** river and reach names to 16 characters.
* ``.f01`` pads the **reach** to 16 but leaves the **river name unpadded**. The
  two files disagree, and matching the geometry's padding in the flow file is
  wrong.
* River stations are left-justified in an 8-character field in both.
* Flow values are right-justified in 8-character fields, all profiles on one line.
* The most downstream cross-section takes reach lengths of ``0,0,0``; every other
  section needs a positive length or HEC-RAS refuses the run.

Run the result with :class:`aquanexus.hecras.runner.RasController`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from aquanexus.logger import get_logger

log = get_logger("hecras.project")

_TEMPLATES = Path(__file__).parent / "templates"

# HEC-RAS writes these files with CRLF line endings.
_EOL = "\r\n"


def _write(path: Path, text: str) -> Path:
    """Write with CRLF preserved exactly.

    ``newline=""`` is essential: without it Python's text layer translates the
    ``\\n`` in an already-CRLF string a second time and emits ``\\r\\r\\n``.
    HEC-RAS accepts such a file, then hangs on a modal dialog instead of
    reporting a parse error - which over COM blocks the caller indefinitely.
    """
    path.write_text(text, encoding="utf-8", newline="")
    return path


@dataclass(frozen=True)
class SteadyFlowProfile:
    """One steady-flow profile: a name and the discharge applied at the reach head."""

    name: str
    discharge: float


def _normalise_eol(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", _EOL)


def _load(name: str) -> str:
    return _TEMPLATES.joinpath(name).read_text(encoding="utf-8")


def write_steady_flow(
    path: Path,
    profiles: list[SteadyFlowProfile],
    river: str,
    reach: str,
    upstream_station: float,
    downstream_slope: float = 0.001,
    title: str = "AquaNexusFlow",
) -> Path:
    """Write a ``.f01`` steady flow file.

    Flow is specified only at ``upstream_station``; HEC-RAS routes it downstream.
    Each profile needs its own boundary-condition block. The downstream boundary
    is normal depth (``Dn Type= 3``), which suits a reach without a controlling
    structure; upstream is left unset, correct for subcritical flow.
    """
    if not profiles:
        raise ValueError("at least one profile is required")
    if downstream_slope <= 0:
        raise ValueError("downstream_slope must be positive for a normal-depth boundary")

    # Note the asymmetry: river unpadded here, reach padded to 16.
    key = f"{river},{reach:<16}"

    lines = [
        f"Flow Title={title}",
        "Program Version=7.00",
        f"Number of Profiles= {len(profiles)} ",
        f"Profile Names={','.join(p.name for p in profiles)}",
        f"River Rch & RM={key},{upstream_station:<8.0f}",
        "".join(f"{p.discharge:8.6g}" for p in profiles),
    ]

    for index in range(1, len(profiles) + 1):
        lines += [
            f"Boundary for River Rch & Prof#={key}, {index} ",
            "Up Type= 0 ",
            "Dn Type= 3 ",
            f"Dn Slope={downstream_slope:g}",
        ]

    lines += [
        "DSS Import StartDate=",
        "DSS Import StartTime=",
        "DSS Import EndDate=",
        "DSS Import EndTime=",
        "DSS Import GetInterval= 0 ",
        "DSS Import Interval=",
        "DSS Import GetPeak= 0 ",
        "DSS Import FillOption= 0 ",
    ]

    return _write(path, _EOL.join(lines) + _EOL)


def write_project(
    directory: Path | str,
    name: str,
    geometry: str,
    profiles: list[SteadyFlowProfile],
    river: str,
    reach: str,
    upstream_station: float,
    downstream_slope: float = 0.001,
    plan_title: str = "Steady",
    short_id: str = "Steady",
) -> Path:
    """Write ``.prj``, ``.g01``, ``.f01`` and ``.p01`` and return the project path.

    ``geometry`` is the .g01 text, normally from
    :func:`aquanexus.hecras.geometry.to_hecras_geometry`.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = directory / name

    geom_path = stem.with_suffix(".g01")
    _write(geom_path, _normalise_eol(geometry))

    write_steady_flow(
        stem.with_suffix(".f01"),
        profiles,
        river=river,
        reach=reach,
        upstream_station=upstream_station,
        downstream_slope=downstream_slope,
    )

    plan = _load("steady_plan.tmpl").format(
        title=plan_title, short_id=short_id, geom="g01", flow="f01"
    )
    _write(stem.with_suffix(".p01"), _normalise_eol(plan))

    prj = _load("project.tmpl").format(title=name)
    prj_path = _write(stem.with_suffix(".prj"), _normalise_eol(prj))

    log.info("wrote HEC-RAS project %s (%d profile(s))", prj_path.name, len(profiles))
    return prj_path
