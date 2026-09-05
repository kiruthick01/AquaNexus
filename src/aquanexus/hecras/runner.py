"""Drive HEC-RAS through its COM automation controller.

HEC-RAS is Windows-only and has no headless mode. It does register a COM server
(``RAS<version>.HECRASController``) which can open a project, run a plan and read
results without a person clicking through the GUI.

Verified against **HEC-RAS 7.0 (April 2026)** on this project's own generated
geometry: the controller reports the expected river, reach and cross-section
stations.

Hard-won details
----------------
* The **plan** selects the geometry, not the project file. A ``.prj`` listing
  ``Geom File=g01`` still yields an empty ``CurrentGeomFile`` and zero rivers
  until a ``.p01`` referencing that geometry exists and is named by
  ``Current Plan``.
* A ``.prj`` referencing a file that does not exist - say ``Unsteady File=u01``
  with no ``.u01`` on disk - makes HEC-RAS raise a **modal dialog**. Over COM
  that hangs the calling process indefinitely with no error. :func:`validate_project`
  checks for dangling references before opening anything.
* Metric input needs ``SI Units`` in the ``.prj``; HEC-RAS otherwise assumes feet
  and the numbers are quietly wrong rather than rejected.
* ``QuitRas()`` does not always terminate ``Ras.exe``. :class:`RasController` is a
  context manager and kills a surviving process on exit.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from aquanexus.config import settings
from aquanexus.logger import get_logger

log = get_logger("hecras.runner")

#: ProgIDs to try, newest first.
PROG_IDS = ("RAS70.HECRASController", "RAS700.HECRASController",
            "RAS65.HECRASController", "RAS641.HECRASController",
            "RAS631.HECRASController", "RAS507.HECRASController")

#: ``.prj`` keys naming a sidecar file, mapped to the extension it must have.
_SIDECAR_KEYS = {
    "Geom File": "",
    "Flow File": "",
    "Unsteady File": "",
    "Plan File": "",
    "Quasi Unsteady File": "",
}


class HecRasError(RuntimeError):
    """HEC-RAS could not be driven to completion."""


def validate_project(prj_path: Path | str) -> list[str]:
    """Return a list of problems that would stop a project opening cleanly.

    Checks for dangling sidecar references, which are the failure mode that
    hangs COM automation on a modal dialog rather than returning an error.
    """
    prj_path = Path(prj_path)
    problems: list[str] = []
    if not prj_path.is_file():
        return [f"project file not found: {prj_path}"]

    text = prj_path.read_text(encoding="utf-8", errors="replace")
    stem = prj_path.with_suffix("")

    for raw in text.splitlines():
        if "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key, value = key.strip(), value.strip()
        if key not in _SIDECAR_KEYS or not value:
            continue
        target = Path(f"{stem}.{value}")
        if not target.is_file():
            problems.append(f"{key}={value} references missing file {target.name}")

    if not re.search(r"^\s*Current Plan\s*=", text, re.MULTILINE):
        problems.append("no 'Current Plan' entry; geometry will not be loaded")

    if not re.search(r"^\s*SI Units\s*$", text, re.MULTILINE):
        problems.append("no 'SI Units' line; HEC-RAS will interpret input as feet")

    return problems


class RasController:
    """Context manager around the HEC-RAS COM controller.

    ``with RasController(prj) as ras:`` opens the project and guarantees the
    process is gone afterwards, including when HEC-RAS ignores ``QuitRas``.
    """

    def __init__(self, project: Path | str, prog_id: str | None = None,
                 validate: bool = True, show: bool = False):
        self.project = Path(project).resolve()
        self.prog_id = prog_id
        self.show = show
        self._ras: Any = None

        if validate:
            problems = validate_project(self.project)
            if problems:
                raise HecRasError(
                    "project would not open cleanly:\n  - " + "\n  - ".join(problems)
                )

    def __enter__(self) -> RasController:
        try:
            import win32com.client
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise HecRasError(
                "pywin32 is required to drive HEC-RAS; pip install '.[hecras]'"
            ) from exc

        candidates = [self.prog_id] if self.prog_id else list(PROG_IDS)
        last: Exception | None = None
        for pid in candidates:
            try:
                self._ras = win32com.client.Dispatch(pid)
                self.prog_id = pid
                break
            except Exception as exc:  # noqa: BLE001 - probing for an installed version
                last = exc
        if self._ras is None:
            raise HecRasError(f"no HEC-RAS COM server found (tried {candidates})") from last

        log.info("%s -> %s", self.prog_id, self.version)
        self._ras.Project_Open(str(self.project))
        if self.show:
            self._ras.ShowRas()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._ras is not None:
            try:
                self._ras.QuitRas()
            except Exception:  # noqa: BLE001 - best effort
                log.debug("QuitRas failed; killing process")
            self._ras = None
        # QuitRas does not reliably terminate Ras.exe.
        subprocess.run(["taskkill", "/F", "/IM", "Ras.exe"],
                       capture_output=True, check=False)

    # -- inspection ---------------------------------------------------------

    @property
    def version(self) -> str:
        return str(self._ras.HECRASVersion())

    @property
    def geometry_file(self) -> str:
        return str(self._ras.CurrentGeomFile())

    def rivers(self) -> list[str]:
        count, names = self._ras.Geometry_GetRivers(0)
        return [n.strip() for n in (names or ())][:count]

    def nodes(self, river: int = 1, reach: int = 1) -> list[str]:
        """River stations in the given reach, as HEC-RAS reports them."""
        result = self._ras.Geometry_GetNodes(river, reach, None, None)
        count = result[2]
        stations = result[3] or ()
        return [s.strip() for s in stations][:count]

    # -- computation --------------------------------------------------------

    def compute(self, timeout_note: str = "") -> tuple[bool, list[str]]:
        """Run the current plan. Returns ``(success, messages)``.

        HEC-RAS reports input problems as a generic "there must have been some
        missing data in the input files", without saying which file or field, so
        the messages are returned verbatim for the caller to surface.
        """
        result = self._ras.Compute_CurrentPlan(None, None, True)
        success = bool(result[0])
        messages = [str(m) for m in (result[2] or ())]
        log.info("compute %s%s", "succeeded" if success else "FAILED",
                 f" ({timeout_note})" if timeout_note else "")
        for m in messages:
            log.debug("  %s", m)
        return success, messages

    def node_output(self, node: int, variable: int, river: int = 1, reach: int = 1,
                    profile: int = 1) -> float:
        """Read one output variable at one node.

        Common variable ids: 2 = water surface elevation, 4 = depth,
        6 = channel velocity.
        """
        return float(self._ras.Output_NodeOutput(river, reach, node, 0, profile, variable))


def find_hecras_exe() -> Path | None:
    """Locate an installed ``Ras.exe``, preferring the configured path."""
    configured = Path(settings.HECRAS_EXE)
    if configured.is_file():
        return configured

    for root in (Path(r"C:\Program Files (x86)\HEC\HEC-RAS"),
                 Path(r"C:\Program Files\HEC\HEC-RAS")):
        if not root.is_dir():
            continue
        versions = sorted((d for d in root.iterdir() if d.is_dir()), reverse=True)
        for version_dir in versions:
            exe = version_dir / "Ras.exe"
            if exe.is_file():
                return exe
    return None
