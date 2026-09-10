"""Tests for the held-out river endpoint.

The endpoint serves a result that also exists as prose in
`docs/HOLDOUT_RIVER.md`, and the same run writes both. The test that matters
here is the one that reads the document back and checks the API agrees with it:
two copies of a number are two chances to be wrong, and the written one is what
a reader will quote.
"""

from __future__ import annotations

import json
import re

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from aquanexus.api.app import app  # noqa: E402
from aquanexus.api.registry import ModelRegistry, registry  # noqa: E402
from aquanexus.config import settings  # noqa: E402

ARTEFACT = settings.PROCESSED_DIR / "holdout_naka.json"
DOCUMENT = settings.ROOT_DIR / "docs" / "HOLDOUT_RIVER.md"

holdout_present = pytest.mark.skipif(
    not ARTEFACT.is_file(),
    reason="no holdout result; run scripts/holdout_river.py",
)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def evidence(client):
    response = client.get("/holdout")
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_missing_result_is_a_503_that_says_what_to_run(client, monkeypatch):
    """Absent evidence must not look like an empty result."""
    monkeypatch.setattr(registry, "holdout", None)
    response = client.get("/holdout")
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "scripts/holdout_river.py" in detail
    assert "docs/HOLDOUT_RIVER.md" in detail


@holdout_present
def test_holdout_loads_without_any_models(tmp_path):
    """The transfer result is a record of an experiment, not a live capability.

    An API with no trained artefacts still has something true to say about what
    happened when this model met a river it had not seen, so the holdout must
    load on the path where the manifest is missing and loading returns early.
    """
    empty = ModelRegistry().load(models_dir=tmp_path)
    assert not empty.ready
    assert empty.error is not None
    assert empty.holdout is not None
    assert empty.holdout["n"] > 0


@holdout_present
def test_unreadable_artefact_degrades_rather_than_raising(tmp_path, monkeypatch):
    broken = tmp_path / "processed"
    broken.mkdir()
    (broken / "holdout_naka.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(settings, "PROCESSED_DIR", broken)
    assert ModelRegistry()._load_holdout() is None


# ---------------------------------------------------------------------------
# The result itself
# ---------------------------------------------------------------------------


@holdout_present
def test_serves_the_transfer_with_its_references(evidence):
    assert evidence["river"] == "Naka"
    assert evidence["river_ja"] == "中川"
    assert "綾瀬川" in evidence["trained_on"]

    labels = [row["label"] for row in evidence["pooled"]]
    assert labels[0] == "Ayase model, unchanged"
    # The floor, the no-model baseline and the ceiling all have to be there:
    # "the model transfers badly" and "this river is hard" are different
    # findings, and only the references separate them.
    assert "mean of this river" in labels
    assert "persistence" in labels
    assert "trained on this river" in labels


@holdout_present
def test_the_split_is_served_and_the_two_regimes_differ(evidence):
    by_label = {row["label"]: row for row in evidence["by_evidence"]}
    inside = next(v for k, v in by_label.items() if k.startswith("inside"))
    outside = next(v for k, v in by_label.items() if k.startswith("outside"))

    assert inside["n"] + outside["n"] == evidence["n"]
    # The finding: inside the training ranges the model works, outside it does
    # not. If this ever stops holding, the headline is no longer true.
    assert inside["r2"] > 0
    assert outside["r2"] < 0
    assert inside["rmse"] < outside["rmse"]


@holdout_present
def test_headline_reports_the_split_and_not_the_pooled_number_alone(evidence):
    """The pooled score is the misleading version of this result."""
    pooled = evidence["pooled"][0]
    inside = next(r for r in evidence["by_evidence"] if r["label"].startswith("inside"))

    assert f"{inside['r2']:+.3f}" in evidence["headline"]
    assert evidence["headline"] != f"{pooled['r2']:+.3f}"
    # ...and the pooled figure is still disclosed, in the caveats.
    assert any(f"{pooled['r2']:+.3f}" in caveat for caveat in evidence["caveats"])


@holdout_present
def test_out_of_range_stations_are_flagged_against_the_served_ranges(evidence, client):
    """The station flag must agree with the ranges /models publishes.

    Two ways to say where the evidence ends is one too many. A client drawing
    the per-station table against `training_ranges` from `/models` would
    otherwise disagree with the flag in the same response body.
    """
    models = client.get("/models")
    if models.status_code != 200:
        pytest.skip("models absent")
    ranges = {m["target"]: m["training_ranges"] for m in models.json()}
    low, high = ranges["dissolved_oxygen"]["discharge"]

    assert evidence["by_station"], "the per-station table is the point"
    for station in evidence["by_station"]:
        inside = low <= station["mean_discharge"] <= high
        assert station["in_training_range"] is inside, station["station"]

    # Both regimes must actually be represented, or the table proves nothing.
    flags = {s["in_training_range"] for s in evidence["by_station"]}
    assert flags == {True, False}


@holdout_present
def test_home_metrics_travel_with_the_transfer(evidence):
    """A transfer score means nothing without the score it is compared to."""
    assert evidence["home_metrics"]["r2"] == pytest.approx(0.394, abs=0.01)
    assert "validation" in evidence["home_metrics"]


# ---------------------------------------------------------------------------
# The API and the document must not drift apart
# ---------------------------------------------------------------------------


@holdout_present
@pytest.mark.skipif(not DOCUMENT.is_file(), reason="document not generated")
def test_served_numbers_match_the_written_document(evidence):
    text = DOCUMENT.read_text(encoding="utf-8")
    rows = document_rows(text)

    for row in evidence["pooled"] + evidence["by_evidence"]:
        assert row["label"] in rows, f"{row['label']} is served but not written"
        written = rows[row["label"]]
        assert written["rmse"] == pytest.approx(row["rmse"], abs=5e-4)
        assert written["r2"] == pytest.approx(row["r2"], abs=5e-4)
        assert written["bias"] == pytest.approx(row["bias"], abs=5e-4)

    for station in evidence["by_station"]:
        written = rows[station["station"]]
        assert written["rmse"] == pytest.approx(station["rmse"], abs=5e-4)
        assert written["r2"] == pytest.approx(station["r2"], abs=5e-4)


def document_rows(text: str) -> dict[str, dict[str, float]]:
    """Every markdown table row in the report, keyed by its label cell.

    Read by column name rather than by position. The three tables carry
    different columns - the pooled and split ones end `RMSE | MAE | R² | bias`
    and the per-station one has two label columns before its numbers - so
    counting in from either end silently reads the wrong column in one of them.
    """
    rows: dict[str, dict[str, float]] = {}
    header: list[str] = []

    for line in text.splitlines():
        if not line.startswith("|"):
            header = []
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if set(line) <= set("|- "):  # the ---|--- separator
            continue
        if not header:
            header = cells
            continue

        # strict=False: a row with the wrong cell count is dropped by the
        # completeness check below rather than raising here.
        row = dict(zip(header, cells, strict=False))
        # The station table repeats a sub-reach across rows; its own name is
        # the key. Everything else is keyed by its first, unnamed column.
        label = row.get("station") or cells[0]
        numbers = {name: as_float(row.get(column, ""))
                   for name, column in (("rmse", "RMSE"), ("r2", "R²"),
                                        ("bias", "bias"))}
        if any(value is None for value in numbers.values()):
            continue
        rows[label] = numbers
    return rows


def as_float(cell: str) -> float | None:
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", cell):
        return None
    return float(cell)


@holdout_present
def test_artefact_and_document_come_from_the_same_run(evidence):
    """The artefact names the document it was written beside."""
    written = json.loads(ARTEFACT.read_text(encoding="utf-8"))
    assert written["document"] == "docs/HOLDOUT_RIVER.md"
    assert written["source"] == "scripts/holdout_river.py"
    assert evidence["generated"] == written["generated"]
