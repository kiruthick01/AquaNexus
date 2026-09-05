# ML Methodology

Working document. Supersedes parts of `../ML_STRATEGY.md`, which was written before
the data audit and assumed a 4-year hourly series on the Yodo.

## Standing caveat

HSI labels are generated, not observed. Any metric reported here describes how well
the model recovers a known function of its own inputs. See the README's
"Scope and honest caveats".

The observed monitoring record does give one genuine check on the label function —
see [Falsification test](#falsification-test-for-the-synthetic-labels) below.

---

## Corrections to ML_STRATEGY.md §3

Implemented in `aquanexus.data.preprocessor`. Three deliberate departures:

### 1. Lagged and rolling features are not produced

§3.2 specifies `*_lag1/6/24` and `*_ma7/ma30` features. Those assume a continuous
hourly chronology. The dataset is a designed experiment over flow space, where rows
are independent samples and "the previous hour" does not exist. Generating them
would manufacture autocorrelation that the physics never produced.

### 2. The thermal stress formula contradicted its own description

§3.2 gives:

```
thermal_stress_index = 1 / (1 + exp(-a * (temp - optimal_temp)))
  Penalty for temp > 25°C or < 10°C
```

That expression is a monotonic sigmoid — it rises with temperature and **never
penalises cold water**, directly contradicting the prose beneath it. The prose is
ecologically correct, so the implementation uses a two-sided Gaussian response about
the optimum (17.5 °C), symmetric in cold and warm departures.

### 3. The oxygen stress formula was unbounded

§3.2 gives:

```
oxygen_stress_index = exp(-b * (optimal_do - do))
```

When DO exceeds the optimum, `(optimal_do - do)` is negative and the expression grows
without limit — it cannot be a stress index. Replaced with a bounded response: 1 at
or below the critical threshold (2 mg/L), 0 at or above ample (8 mg/L), squared in
between because the ecological cost of losing oxygen accelerates as levels fall.

### Also: DO saturation is computed, not assumed

§3.1 treats `optimal_do` as a constant 10 mg/L. Saturation is strongly temperature
dependent — 14.6 mg/L at 0 °C against 8.3 at 25 °C — so a fixed reference would
misread ordinary summer warming as oxygen depletion. `do_saturation()` implements the
Benson–Krause equation (APHA 4500-O) with an elevation pressure correction, and is
tested against the published saturation table to ±0.03 mg/L.

---

## Observed design space (Ayase, FY2022–24, n=216)

Derived with `observed_ranges()` from the real monitoring record. These bounds define
the simulation sweep, replacing the guessed ranges in §3.1:

| Variable | q0.01 | q0.99 | ML_STRATEGY §3.1 said | 
|---|---|---|---|
| water_temp (°C) | 5.6 | 32.5 | 5–35 ✓ |
| discharge (m³/s) | 0.25 | 64.2 | **50–500 ✗** |
| dissolved_oxygen (mg/L) | 3.4 | 12.9 | 0–14 ✓ |
| bod (mg/L) | 0.8 | 8.7 | 0–10 ✓ |
| nitrogen_total (mg/L) | 1.7 | 6.2 | **0–5 ✗** |
| phosphorus_total (mg/L) | 0.14 | 0.38 | 0–1 ✓ |
| suspended_solids (mg/L) | 4.0 | 41.9 | **0–500 ✗** |

The discharge range is the significant one: the plan's lower bound of 50 m³/s is
above the Ayase's 99th percentile. Simulating that range would have produced a
dataset with almost no overlap with the real river.

Quantiles rather than min/max, so one fouled sample cannot stretch the design space.

---

## Falsification test for the synthetic labels

The observed record carries a real, independent relationship:

```
DO above 25 °C:  5.13 mg/L  (n=59)
DO below 15 °C:  8.74 mg/L  (n=72)
```

Worst observed condition: **31.2 °C at 3.1 mg/L DO — 42% saturation**
(57綾瀬川合流点前, 2024-09-18). Best: 17.8 °C at 84% saturation, zero computed stress.

**If the generated HSI does not degrade across that gradient, the label function is
wrong.** This does not make the labels real, but it makes them refutable, which is a
materially stronger position than the original plan had.

Note the record also contains genuine **supersaturation** — DO up to 17.0 mg/L, and
106% saturation on 2023-04-13. This is real (spring algal photosynthesis), not error.
`do_deficit` is deliberately allowed to go negative rather than being clipped, since
supersaturation is itself a eutrophication signal.

---

## Validation splits — revised

§5's temporal split does not apply; there is no chronology to split on.

- **Held-out flow space** (primary) — train on the interior of the design grid, test
  on an unseen corner such as high discharge with high temperature.
- **Held-out reach** (secondary) — still valid. Naka is reserved for this; it has
  contiguous point-cloud coverage and its own monitoring stations.
- **Held-out extremes** (tertiary) — reframed from "event-based": the extreme
  conditions within the design grid.

## Benchmarks — to re-derive

The R² 0.80–0.88 in §9 was set against the hourly-series design and should not be
carried over unexamined. With synthetic labels these numbers measure function
recovery, not ecological skill, and the figure should be reported that way.
