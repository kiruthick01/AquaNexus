# Contributing

This is a single-author proof of concept, so there is no review process to
describe. What follows is how to run it, and the conventions a change is
expected to follow — mostly so that future me, or anyone picking it up, does not
have to reverse-engineer them from the diff.

## Getting it running

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows; source .venv/bin/activate elsewhere
pip install -e ".[ml,api,dev]"                      # add ",geo" for point clouds
pytest                                              # 376 tests
ruff check src tests scripts

cd frontend && npm install && npm test              # 22 tests
```

The suite passes without any data: tests that need trained artefacts skip
themselves and say why. To get the artefacts, see the rebuild sequence in
`README.md` — the data is 9.5 GB of point cloud and is gitignored.

HEC-RAS 7.x on Windows is needed only to *generate* hydraulics
(`scripts/build_geometry.py`, `scripts/run_hecras.py`). Everything else runs
anywhere.

## What a change is expected to do

**Say why in the commit message, not what.** The diff already shows what
changed. The message is for the reader who wants to know which constraint forced
it — an API that removed a field, a formula that contradicted its own
description, a test that passed on nothing.

**Keep provenance attached to numbers.** One of the two served models is trained
on labels this repository generated. Any code path that produces a prediction
also carries its `labels` and `caveats`, and the dashboard renders them in the
same card as the number. A change that separates them is a regression, and
`frontend/src/test/provenance.test.tsx` is there to fail when one does.

**Report results as found.** Several published numbers in this project got
*worse* when a defect was fixed — removing four bad cross-sections cost 0.05 R²,
and correcting a SHAP background revealed an explanation endpoint that had been
returning zeros. Both are documented at their new values. Choosing between
implementations by which one scores better is the failure mode this project is
written against.

**Retract rather than quietly amend.** When a finding does not survive
re-analysis, say so where it was published. `docs/ML_METHODOLOGY.md` carries a
retraction of a feature interaction for this reason.

**Comments explain decisions, not mechanics.** `# increment i` is noise;
`# XGBoost 2.x moved this into the constructor, and passing it to fit() raises`
is the reason somebody will need in a year.

## Testing conventions

- Tests are named for the behaviour they protect, not the function they call:
  `test_cache_hits_do_not_spend_the_rate_limit`, not `test_cache_2`.
- A regression test's docstring says what broke and how it looked when it did.
- Assert that a result is *non-degenerate*, not only self-consistent. A test
  asserting SHAP contributions summed correctly passed for days while every
  contribution was zero.
- Drive time with a fake clock. Tests that sleep get deleted.

## Layout

| Path | Holds |
|---|---|
| `src/aquanexus/` | the package: data, hecras, ml, api |
| `scripts/` | entry points that produce artefacts, each runnable alone |
| `tests/` | pytest, mirroring the package plus deployment invariants |
| `notebooks/` | executable narrative, committed unexecuted |
| `docs/` | methodology, data sources, API, deployment, sensitivity studies |
| `frontend/` | React dashboard over the API |

Generated artefacts (`data/`, `frontend/dist/`) are gitignored. The scripts that
produce them are not.
