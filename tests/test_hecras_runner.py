"""Runner tests.

``validate_project`` is tested directly, since it exists to catch the exact
conditions that hang COM automation. The controller itself needs an installed
HEC-RAS, so those tests skip when it is absent.
"""

from pathlib import Path

import pytest

from aquanexus.hecras.runner import (
    HecRasError,
    RasController,
    find_hecras_exe,
    validate_project,
)

GOOD_PRJ = """Proj Title=Ayase
Current Plan=p01
Default Exp/Contr=0.3,0.1
SI Units
Geom File=g01
Flow File=f01
Plan File=p01
"""


def make_project(tmp_path: Path, prj_text: str = GOOD_PRJ, sidecars=("g01", "f01", "p01")):
    prj = tmp_path / "Ayase.prj"
    prj.write_text(prj_text, encoding="utf-8")
    for ext in sidecars:
        (tmp_path / f"Ayase.{ext}").write_text("placeholder\n", encoding="utf-8")
    return prj


# ---------------------------------------------------------------------------
# validate_project
# ---------------------------------------------------------------------------


def test_valid_project_has_no_problems(tmp_path):
    assert validate_project(make_project(tmp_path)) == []


def test_dangling_sidecar_is_reported(tmp_path):
    """The failure that hangs COM on a modal dialog instead of erroring."""
    prj = make_project(tmp_path, GOOD_PRJ + "Unsteady File=u01\n")
    problems = validate_project(prj)
    assert any("u01" in p and "missing" in p for p in problems)


def test_missing_geometry_file_reported(tmp_path):
    prj = make_project(tmp_path, sidecars=("f01", "p01"))
    assert any("Geom File" in p for p in validate_project(prj))


def test_missing_current_plan_reported(tmp_path):
    text = GOOD_PRJ.replace("Current Plan=p01\n", "")
    assert any("Current Plan" in p for p in validate_project(make_project(tmp_path, text)))


def test_missing_si_units_reported(tmp_path):
    """Without SI Units the numbers are wrong rather than rejected."""
    text = GOOD_PRJ.replace("SI Units\n", "")
    assert any("SI Units" in p for p in validate_project(make_project(tmp_path, text)))


def test_absent_project_file(tmp_path):
    assert validate_project(tmp_path / "nope.prj") == [
        f"project file not found: {tmp_path / 'nope.prj'}"
    ]


def test_controller_refuses_an_invalid_project(tmp_path):
    """Construction must fail before anything can open a dialog."""
    prj = make_project(tmp_path, GOOD_PRJ + "Unsteady File=u01\n")
    with pytest.raises(HecRasError, match="would not open cleanly"):
        RasController(prj)


def test_validation_can_be_skipped(tmp_path):
    prj = make_project(tmp_path, GOOD_PRJ + "Unsteady File=u01\n")
    RasController(prj, validate=False)  # constructs without opening


# ---------------------------------------------------------------------------
# Installed HEC-RAS
# ---------------------------------------------------------------------------

hecras = pytest.mark.skipif(find_hecras_exe() is None, reason="HEC-RAS not installed")


@hecras
def test_finds_installed_executable():
    exe = find_hecras_exe()
    assert exe.is_file() and exe.name.lower() == "ras.exe"
