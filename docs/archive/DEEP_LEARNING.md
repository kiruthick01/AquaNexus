# Deep Learning (Phase 5)

## Status: MLP implemented and evaluated; LSTM implemented as infrastructure only

## Architecture

`src/aquanexus/ml/deep.py`, PyTorch (`pyproject.toml` `deep` extra, not a
core dependency). Deliberately small - at n~100-140 real training rows, a
network with more parameters than training points is in the same overfitting
regime `ML_METHODOLOGY.md` already documented for XGBoost on this dataset.

**MLP**: two hidden layers (16, 8 units), ReLU, dropout 0.2, trained with
Adam (learning rate 1e-2, weight decay 1e-3), early stopping on a held-out
validation slice (patience 20 epochs, max 300), checkpointing the
best-validation-loss weights rather than the last epoch's. Median imputation
and standardization are fit on the training fold only, mirroring the linear
pipeline's rule in `ml/models.py`.

**LSTM**: single-layer, hidden size 8, otherwise the same training loop.
**Not run against real AquaNexus data** - see Limitations.

## Reproducibility

`set_seed()` seeds Python's `random`, NumPy, and Torch together before every
fit. `tests/test_deep.py::test_mlp_is_reproducible_with_same_seed` asserts
two independent fits with the same seed produce bit-identical predictions,
not merely similar ones.

## Validation protocol

Same leave-one-station-out protocol as every other model in this project
(`ml.validator.ModelValidator`, `docs/VALIDATION.md`). For each held-out
station, a further random 20% validation slice is carved from the three
remaining training stations purely for early stopping - it never touches the
held-out station, so the reported score is a genuine out-of-sample number,
not one where the model was allowed to peek at its own test data to decide
when to stop training.

## Results (real data, `scripts/phase5_deep_learning_experiment.py`)

| Model | n | RMSE | MAE | R² | bias |
|---|---|---|---|---|---|
| Ridge (linear) | 138 | 1.785 | 1.274 | 0.394 | -0.279 |
| Persistence | 134 | 1.818 | 1.213 | 0.385 | -0.106 |
| Random Forest | 138 | 1.884 | 1.425 | 0.325 | 0.023 |
| XGBoost | 138 | 1.909 | 1.408 | 0.307 | -0.101 |
| Mean (floor) | 138 | 2.294 | 1.779 | 0.000 | 0.000 |
| **MLP** | 138 | **2.358** | **1.904** | **-0.057** | -0.056 |

**MLP loses to every other model, including the constant-mean floor.** An R²
below 0.000 means the MLP's predictions are worse than simply always
predicting the training mean - not a marginal loss, a clear one.

**Training cost**: 321 parameters, mean best epoch 87 (of up to 300), mean
0.89s training time per fold on CPU, 1.3ms inference for all 138 rows.
Compute cost was never the constraint here - data was.

## Interpretation

The audit's prediction (`docs/ML_ROADMAP.md` §4, written before this
experiment ran) was that an MLP would be "technically runnable... but should
be expected to lose to Ridge, consistent with the existing finding that
XGBoost already overfits at this sample size." That prediction held, and more
strongly than stated: the MLP does not just lose to Ridge, it loses to
predicting a constant. With only ~90-120 training rows per fold and 10
features, a 321-parameter network - itself already the smaller end of what
"deep learning" usually means - still has far more capacity than a linear
model at this sample size, and no amount of dropout/weight-decay/early-
stopping regularisation fully closes that gap on a dataset this small. This
is not a claim that MLPs cannot work for dissolved-oxygen prediction in
general; it is a specific, measured claim about this dataset's size.

## Limitations

- **LSTM was not run against real data.** Phase 1 (`docs/EXPERIMENTS.md`
  EXP-001) found zero usable lag-complete forecasting rows at the current
  sampling density - there is no real sequence for an LSTM to train on. It
  is implemented and tested against synthetic sequences only
  (`tests/test_deep.py`), so it does not need to be built from scratch if
  Phase 1's data blocker clears.
- A single architecture (16, 8 hidden units) was evaluated, not a
  hyperparameter search - tuning architecture size against the same n=138
  evaluation set used to report this result would leak, and a larger search
  would not change the qualitative finding (more capacity is the wrong
  direction to move at this sample size, not an under-explored one).
- Random-forest-style per-tree disagreement (`HabitatPredictor.
  predict_with_uncertainty`) has no analogue implemented for the MLP; if MLP
  uncertainty were needed, `ml/uncertainty.py`'s `BootstrapIntervalModel`
  interface would apply to it directly, but this was not attempted since the
  point prediction itself is not competitive.

## Conclusion

Deep learning does not outperform classical ML on this dataset, and the gap
is not close. The canonical point-prediction model for dissolved oxygen
remains Ridge (`ML_METHODOLOGY.md`), unchanged by this phase.
