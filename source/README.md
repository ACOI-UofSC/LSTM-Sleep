# source/

**Pipeline Step 3.** Loads extracted features, runs leave-one-subject-out (LOSO) cross-validation using the LocalGlobalLSTM, and writes per-subject prediction files, model checkpoints, and feature importance scores.

## Entry point

```bash
cd source
python analysis_runner_weighted_split_torch.py [--output-dir PATH] [--no-loso] [--seed 42]
```

| Argument | Default | Description |
|---|---|---|
| `--output-dir` | `../outputs/agv_loso` | Root directory for all outputs |
| `--no-loso` | False | Use 10-fold CV instead of LOSO |
| `--seed` | 42 | Random seed for reproducibility |

## Files

| File | Purpose |
|---|---|
| `analysis_runner_weighted_split_torch.py` | **Entry point** — orchestrates training, LOSO, output saving |
| `constants.py` | Centralised path configuration (edit to change data locations) |
| `utils.py` | Defines the 4-channel AGV feature set used by the model |
| `sleep_stage.py` | `SleepStage` enum: `wake=0`, `n1=1`, `n2=2`, `n3=3`, `n4=4`, `rem=5` |

## Sub-packages

| Package | Purpose |
|---|---|
| `analysis/` | Model architecture, training loop, evaluation, and performance storage |

## Configuration

All data paths are defined in `constants.py`. The key variables are:

```python
DEVICE = 'agv'
INPUT_ROOT = Path('../data/data_processed/')
PSG_FILE_PATH     = INPUT_ROOT / DEVICE / 'labels/'
CROPPED_FILE_PATH = INPUT_ROOT / DEVICE / 'cropped/'
FEATURE_FILE_PATH = INPUT_ROOT / DEVICE / 'features/'
MOTION_FILE_PATH  = INPUT_ROOT / DEVICE / 'motion/'
```

Change `INPUT_ROOT` if your data lives elsewhere. All other paths derive from it automatically.

## Output structure

```
<output_dir>/LocalGlobalLSTM/motion_xfft1_30+motion_yfft1_30+motion_zfft1_30+motion_vmfft1_30/
    predictions/
        ML_pred_<subject_id>_lstm.csv     # per-subject epoch predictions
    models/
        fold_<subject_id>.pt              # best-val model checkpoint per LOSO fold
    feature_importance.csv               # permutation importance per fold + grand mean
```

## Device detection

The runner auto-selects the best available PyTorch device:

```python
MPS  (Apple Silicon)  →  if torch.backends.mps.is_available()
CUDA (NVIDIA GPU)     →  elif torch.cuda.is_available()
CPU                   →  fallback
```
