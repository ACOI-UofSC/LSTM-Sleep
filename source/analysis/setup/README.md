# source/analysis/setup/

Data preparation and cross-validation split utilities: loads per-subject feature files, encodes labels, and generates train/test fold assignments.

## Files

| File | Purpose |
|---|---|
| `subject_builder.py` | Loads all four FFT feature arrays and labels for every subject into a dictionary |
| `subject.py` | `Subject` dataclass: `feature_dictionary` + `labeled_sleep` |
| `feature_type.py` | `FeatureType` enum — unique identifier for every feature channel type |
| `sleep_label.py` | `SleepWakeLabel` enum: **`wake=0`, `sleep=1`** |
| `sleep_labeler.py` | Converts raw PSG stage values to binary (0=Wake, 1=Sleep) or multi-class labels |
| `train_test_splitter.py` | Generates LOSO or k-fold CV splits |
| `data_split.py` | `DataSplit` dataclass: `training_set` + `testing_set` subject ID lists |
| `attributed_classifier.py` | Wraps a model with a name string for reporting |
| `feature_set_service.py` | Utility to look up features from subject dictionaries |

## Label encoding

The `sleep_labeler.py` rule is simple:

```python
label = 0  if raw_stage == 0    (SleepStage.wake)
label = 1  if raw_stage > 0     (any of N1, N2, N3, N4, REM)
```

This is defined by `SleepWakeLabel.wake = 0` and `SleepWakeLabel.sleep = 1` in `sleep_label.py` and verified by unit tests. The same encoding applies to both ground-truth labels (`label_n`) and the subject's stored features.

## Cross-validation splits (`train_test_splitter.py`)

```python
# Leave-one-subject-out (used in this fork)
splits = TrainTestSplitter.leave_one_out(subject_ids)
# → list of DataSplit(training_set=[all but one], testing_set=[one])

# k-fold (fallback, use with --no-loso)
splits = TrainTestSplitter.by_number(subject_ids, n_splits=10)
```

> **Note:** The original codebase also contained `by_number_age()` which sorted subjects by age before stratifying. This was removed in this fork because age is not available in the AGV dataset and was not used as a model feature.

## Feature dictionary structure

Each `Subject` object contains:

```python
subject.feature_dictionary = {
    FeatureType.motion_xfft1_30:  np.ndarray,  # shape (N_epochs, 30)
    FeatureType.motion_yfft1_30:  np.ndarray,
    FeatureType.motion_zfft1_30:  np.ndarray,
    FeatureType.motion_vmfft1_30: np.ndarray,
}
subject.labeled_sleep = np.ndarray  # shape (N_epochs,)  values in {0, 1}
```

The `ClassifierInputBuilder` stacks these into `(N_epochs, 30, 4)` tensors for the model.
