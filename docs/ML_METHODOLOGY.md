# ML Methodology

> Placeholder — filled in during Phase 2. The substance currently lives in
> `../ML_STRATEGY.md`; sections 5 (splits) and 9 (benchmarks) need revision because the
> dataset is now a designed experiment over flow space rather than a 4-year hourly series.

## To revise from ML_STRATEGY.md

- **§5.1 Temporal split** — no longer applicable; there is no chronology to split on.
  Replace with a held-out region of flow space (e.g. unseen high-discharge corner).
- **§5.2 Spatial split** — still valid; hold out reaches.
- **§5.3 Event-based split** — reframe as held-out extreme conditions within the design grid.
- **§9 Benchmarks** — the quoted R² 0.80–0.88 was set against the hourly-series design and
  should be re-derived. Note that with synthetic labels these numbers measure function
  recovery, not ecological skill.

## Standing caveat

HSI labels are generated, not observed. Any metric reported here describes how well the
model recovers a known function of its own inputs. See README "Scope and honest caveats".
