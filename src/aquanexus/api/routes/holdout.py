"""The held-out river: what happened when the model met a catchment it had
never seen.

Every other number this API reports about generalisation is *within* the Ayase
— four stations held out one at a time, sharing a channel, a catchment, a sewer
network and a sampling programme. That is a real test and a weak one. The Naka
(中川) was reserved during the data audit and never touched until the model was
finished, and this endpoint serves the result.

It is also the only place the out-of-range flag on `/predict` is quantified.
The flag has been attached to every prediction since the API was first served;
this says what it is worth — on the wrong side of it the model is worse than
predicting the river's mean, and on the right side it is about as good as it is
at home.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from aquanexus.api.registry import registry
from aquanexus.api.schemas import HoldoutResponse, TransferScore, TransferStation

router = APIRouter(tags=["meta"])

#: Which pooled row is the transfer itself rather than a reference computed on
#: the holdout river. `scripts/holdout_river.py` writes it first.
TRANSFER_ROW = 0


@router.get("/holdout", response_model=HoldoutResponse,
            summary="The model on a river it was never trained on")
def holdout() -> HoldoutResponse:
    """Serve the transfer result, split by whether the model has evidence.

    The pooled score is included, but it is not the answer: it averages a model
    working with the same model extrapolating, and reporting it alone would be
    the misleading version of this result. `headline` and `by_evidence` carry
    the split, and both are derived from the run that produced the numbers.
    """
    evidence = registry.holdout
    if not evidence:
        raise HTTPException(
            status_code=503,
            detail="no holdout result available; run scripts/holdout_river.py. "
                   "The written version is in docs/HOLDOUT_RIVER.md.",
        )

    river = evidence.get("river", {})
    return HoldoutResponse(
        river=river.get("name", "unknown"),
        river_ja=river.get("name_ja", ""),
        trained_on=evidence.get("trained_on", {}).get("river", ""),
        n=int(evidence.get("n", 0)),
        n_stations=int(evidence.get("n_stations", 0)),
        generated=evidence.get("generated", ""),
        headline=evidence.get("headline", ""),
        pooled=[_score(row, row.get("model", "")) for row in evidence.get("pooled", [])],
        by_evidence=[_score(row, row.get("subset", ""))
                     for row in evidence.get("by_evidence", [])],
        by_station=[TransferStation(**row) for row in evidence.get("by_station", [])],
        home_metrics=dict(evidence.get("trained_on", {}).get("metrics", {})),
        document=evidence.get("document", "docs/HOLDOUT_RIVER.md"),
        caveats=list(evidence.get("caveats", [])),
    )


def _score(row: dict, label: str) -> TransferScore:
    """One scored row. The pooled and split tables carry the same metrics under
    different labels, so they are narrowed to a single shape here rather than
    served as two schemas that differ only in the name of one field."""
    return TransferScore(
        label=label, n=int(row["n"]), rmse=float(row["rmse"]),
        mae=None if row.get("mae") is None else float(row["mae"]),
        r2=float(row["r2"]), bias=float(row["bias"]),
    )
