# Transfer Learning / Domain Adaptation (Phase 6)

## Terminology, precisely

The spec this project follows is explicit that these four are not
interchangeable, and conflating them would misreport what was actually done:

| Term | Target-domain information used | What it means here |
|---|---|---|
| **Zero-shot transfer** | None | The Ayase Ridge model, completely unchanged, applied to the Naka. Already measured (`scripts/holdout_river.py`, `docs/HOLDOUT_RIVER.md`): pooled R² -0.081. |
| **Domain alignment** | Target *feature* statistics only, no labels | `domain_aligned_predict` (`ml/transfer.py`): re-centres inputs on the target's own mean/std before they reach the frozen source coefficients. |
| **Fine-tuning** | A small labelled target slice | `fine_tune_ridge`: refits coefficients on that slice, shrunk toward the source model's own coefficients (L2-SP-style), not fit independently from scratch. |
| **Frozen-feature-extractor + target head** | The same small labelled slice | `FrozenHeadAdapter`: retrains only the final linear layer of the Phase 5 MLP, freezing every earlier layer. |

**Ordinary retraining is none of these** and is not reported as transfer
learning anywhere in this document; "trained on this river" in
`docs/HOLDOUT_RIVER.md` is the ceiling reference, explicitly labelled as
such, not a transfer method.

## Protocol

Source: the full Ayase dataset (138 rows, 4 stations) - identical to every
other phase's training data. Target: the Naka dataset (192 rows, 5
stations), split once, by row, with a fixed seed:

- **Adaptation slice** (15%, 29 rows): the only target data any adaptation
  method may see, of any kind (features and/or labels depending on the
  method).
- **Held-out test set** (85%, 163 rows): scored on by every method,
  including zero-shot - recomputed on this exact subset so all four methods
  are compared on identical rows, not on the full 192-row set the standing
  zero-shot number uses.

Row-level (not station-level) random splitting is used here for the same
reason it was used in `ml/uncertainty.py`'s `SplitConformalModel`: the Naka
dataset carries one row per real observation, not the per-observation row
duplication `ml/splits.py` warns about, so a random split does not
reintroduce that leakage.

## Results (real data, `scripts/phase6_transfer_learning_experiment.py`)

| Method | n | RMSE | MAE | R² | bias |
|---|---|---|---|---|---|
| Zero-shot (Ridge, unchanged) | 163 | 2.075 | 1.496 | -0.133 | -1.197 |
| Domain-aligned (no target labels) | 163 | 2.034 | 1.496 | -0.089 | -1.307 |
| Fine-tuned, shrinkage=1 | 163 | 4.834 | 3.956 | -5.149 | -1.300 |
| Fine-tuned, shrinkage=10 | 163 | 5.355 | 4.461 | -6.548 | -1.484 |
| Fine-tuned, shrinkage=100 | 163 | 3.749 | 3.057 | -2.700 | -1.182 |
| Fine-tuned, shrinkage=1000 | 163 | **1.544** | **1.100** | **0.372** | -0.091 |
| Fine-tuned, shrinkage=10000 | 163 | 1.674 | 1.282 | 0.263 | 0.497 |
| Frozen-head (MLP) | 163 | 1.573 | 1.158 | 0.349 | 0.433 |

The shrinkage sweep (1, 10, 100, 1000, 10000) was run in full and is reported
in full here - no value was chosen after seeing held-out performance and
reported alone as "the" fine-tuning result; the shape of the whole curve is
the finding.

## Interpretation

**Domain alignment gave a small, real improvement over zero-shot** (R² -0.089
vs -0.133, RMSE 2.034 vs 2.075) using no target labels at all - just target
feature statistics. Consistent with `HOLDOUT_RIVER.md`'s documented finding
that the Naka reads warmer/cleaner throughout: re-centring on the target's
own distribution partially corrects for that level shift.

**Fine-tuning is catastrophic at low-to-moderate shrinkage (1, 10, 100) and
the best method overall at high shrinkage (1000).** This is not a monotonic
"more shrinkage is always better" story either: shrinkage=1000 (R² 0.372)
clearly beats shrinkage=10000 (R² 0.263), which is already pulling back
toward zero-shot's -0.133. With only 29 adaptation rows and 10 standardised
features, $X^TX$'s diagonal is of order 29; a shrinkage of 1-100 is too weak
relative to that scale to meaningfully anchor the fit to the source
coefficients, so the result is close to an independent least-squares fit on
29 noisy points - exactly the small-sample overfitting regularisation exists
to prevent. Only once shrinkage is large enough (1000, roughly 30-1000x
$X^TX$'s scale) does the fit become a genuine, small, useful correction to
the source coefficients rather than an under-anchored refit. This is the
concrete answer to "when does transfer fail": not "fine-tuning is risky", but
specifically "fine-tuning with a shrinkage strength not scaled to the
adaptation sample size is risky" - and when scaled correctly, it was this
experiment's single best-performing method.

**Frozen-head adaptation of the MLP** (R² 0.349) performs almost identically
to the best fine-tuned Ridge (R² 0.372), despite its base MLP losing badly to
Ridge in-domain (`docs/DEEP_LEARNING.md`, R² -0.057 on the Ayase itself).
Adapting only the final layer acts as strong implicit regularisation - the
same shape of protection that adequately-shrunk fine-tuning needed an
explicit hyperparameter to achieve.

**The common thread across all four methods**: every method that performed
well (domain alignment, high-shrinkage fine-tuning, frozen-head) shares
strong regularisation of the target-domain adaptation relative to how little
target data exists; every method that performed badly (low/moderate-shrinkage
fine-tuning) under-regularised that adaptation.

## Limitations

- All results are from a single random seed and a single 15/85 adaptation/
  test split; neither the adaptation slice's rows nor its size was varied.
- The shrinkage sweep is a diagnostic explaining the full sensitivity curve,
  not a hyperparameter search followed by reporting only the winning value in
  isolation - the whole table above is the result, not shrinkage=1000 alone.
- Frozen-head adaptation's strong result rests on a weak base model (MLP)
  and a very small adaptation slice; it should not be read as "neural
  transfer beats linear transfer" in general from one run.
- The domain-alignment improvement is small relative to the likely sampling
  noise in a 163-row test set; it is directionally consistent with the
  documented Naka bias, not large enough to call decisive on its own.

## Conclusion

Whether transfer helps or hurts here is governed by how strongly any
target-domain adaptation is regularised relative to how little target data
backs it, not by which named technique is used. Under-regularised fine-tuning
was this experiment's worst outcome, worse than doing nothing (zero-shot);
adequately-regularised fine-tuning, domain alignment, and frozen-head
adaptation were all real improvements over zero-shot, with adequately-shrunk
fine-tuning the strongest of the four methods tried.
