# source/analysis/classification/

Orchestrates the full cross-validation loop: splits subjects into train/test folds, runs the training loop for each fold, computes permutation feature importance, and obtain results.

## Files

| File | Purpose |
|---|---|
| `classifier_summary_builder.py` | Top-level builder: creates LOSO or k-fold splits and runs all folds |
| `classifier_service.py` | Runs one fold: data loading → training → testing → importance → checkpoint |
| `classifier_input_builder.py` | Assembles feature arrays from the subject dictionary into model inputs |
| `classifier_summary.py` | Container: holds all fold results keyed by feature set |

## Cross-validation (`classifier_summary_builder.py`)

Two modes are available:

| Method | When to use |
|---|---|
| `build_leave_one_out()` | **Default.** LOSO with 65 subjects → 65 folds. Each fold trains on 64, tests on 1. |
| `build_leave_multiple_out()` | k-fold CV (e.g., 10-fold). Uses `--no-loso` flag. |

## Feature importance (`classifier_service.py`)

After each fold's training, permutation importance is estimated for each of the 4 feature channels:

1. Compute baseline accuracy on the test subject.
2. For each channel f in [x_fft, y_fft, z_fft, vm_fft]:
   - Shuffle that channel's values **across epochs** (breaking the temporal signal while preserving statistics).
   - Re-evaluate accuracy.
   - Importance = baseline accuracy − permuted accuracy.
3. Repeat 3 times and average.

**Note:** Probably doesn't needed. The result wasn't as expected. The section was kept for the legacy purpose.

## Model checkpointing

After `Trainer.fit()`, the best-validation-accuracy model state dict is deep-copied into `RawPerformance.model_state_dict`. The analysis runner then saves it to `models/fold_<subject>.pt`. This checkpoint can be loaded and applied to new subjects without re-training:

## `_FEATURE_NAMES`

The module-level constant `_FEATURE_NAMES` lists the four channels in the order they appear in the model's last dimension:

```python
_FEATURE_NAMES = ['motion_xfft1_30', 'motion_yfft1_30',
                  'motion_zfft1_30', 'motion_vmfft1_30']
```

This order must match `utils.get_lstm_feature_fft30_sets()`.
