# source/analysis/

Core machine learning components: the LSTM model architecture, PyTorch training loop, classification pipeline, and per-subject performance storage.

## Files

| File | Purpose |
|---|---|
| `model.py` | `LocalGlobalLSTM` architecture and `Trainer` class |
| `dataset.py` | `TimeSeriesDataset` (PyTorch Dataset) and `collate_fn` for variable-length nights |
| `clac_metric.py` | Helper metrics (accuracy, sensitivity, specificity) |

## Sub-packages

| Package | Purpose |
|---|---|
| `classification/` | Orchestrates training runs across data splits; computes feature importance |
| `performance/` | Stores per-fold results (predictions, probabilities, model state, importance) |
| `setup/` | Subject loading, feature type definitions, label encoding, CV splits |

## Model architecture (`model.py`)

The `LocalGlobalLSTM` is a two-level bidirectional LSTM:

```
Input: (batch, N_epochs, 30_fft_bins, 4_feature_types)
           │
           ▼  Local Bidirectional LSTM  (processes each epoch independently)
(batch × N_epochs, local_hidden × 2)
           │  → fully connected → (batch, N_epochs, local_hidden)
           ▼  Global Bidirectional LSTM  (processes the full night sequence)
(batch, N_epochs, global_hidden × 2)
           │  → linear → softmax
           ▼
(batch, N_epochs, 2)   ← per-epoch sleep/wake probabilities
```

The local LSTM captures the spectral signature of a single 30-second epoch; the global LSTM models how that signature evolves across the full night. Class-imbalance is handled via balanced class weights passed to `CrossEntropyLoss`.

## Variable-length sequences

Sleep recordings vary in length across subjects. `collate_fn` pads shorter sequences with −1 labels (masked out during loss and evaluation) so that a batch can contain subjects with different numbers of epochs.

## Training (`Trainer`)

- Optimizer: Adam (lr = 0.0001)
- Gradient clipping: max_norm = 1
- Early stopping: best model by validation accuracy is saved and restored after training
- Epochs: 300 (configurable via `Trainer.__init__`)
