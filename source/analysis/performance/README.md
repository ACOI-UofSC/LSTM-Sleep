# source/analysis/performance/

Stores all outputs from a single LOSO fold in one object, which is then used by the analysis runner to write prediction CSVs, model checkpoints, and feature importance files.

## Files

| File | Purpose |
|---|---|
| `raw_performance.py` | `RawPerformance` dataclass — holds every output produced by one training fold |

## `RawPerformance` fields

| Field | Type | Description |
|---|---|---|
| `true_labels` | list of arrays | Ground-truth binary labels per subject (0=Wake, 1=Sleep) |
| `class_probabilities` | list of arrays | Softmax output — shape `(N_epochs, 2)` per subject |
| `predicted_labels` | list of arrays | `argmax(class_probabilities)` — 0 or 1 per epoch |
| `subject` | list | Subject ID(s) in the test set for this fold |
| `feature_importance` | dict or None | `{feature_name: mean_accuracy_drop}` from permutation importance |
| `model_state_dict` | dict or None | `copy.deepcopy(trainer.best_model)` — PyTorch state dict |

## Notes on `model_state_dict`

The state dict is a deep copy made immediately after `Trainer.fit()` returns, before the next fold starts training. This is important because the model object is shared across folds (warm start), so without a deep copy the dict would be overwritten by subsequent folds.

To reload a saved checkpoint:

```python
import torch
from source.analysis.model import LocalGlobalLSTM

model = LocalGlobalLSTM(feature_dim=4, local_steps=30, n_class=2)
model.load_state_dict(torch.load('models/fold_subject_001.pt', map_location='cpu'))
model.eval()
```

## Notes on `feature_importance`

Importance values are **accuracy drops**: a larger positive number means the model relies more on that feature channel. A value near zero means the model is insensitive to temporal shuffling of that channel.

The `feature_importance.csv` written by `analysis_runner_weighted_split_torch.py` includes one row per subject per feature, plus a `__mean__` summary row averaging across all LOSO folds.
