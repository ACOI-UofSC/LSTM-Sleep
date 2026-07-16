# preprocessing/

**Pipeline Step 2.** Crops raw signals to their overlapping time window, then extracts per-epoch FFT features and saves them as intermediate `.out` files.

## Entry point

```bash
cd preprocessing
python preprocessing_feature.py
```

This script reads subject IDs from `../data/data_processed/agv_ids.csv` (written by `data_ingestion.py`) and runs the full crop + feature pipeline for each subject.

## Scripts

| Script | Purpose |
|---|---|
| `preprocessing_feature.py` | Entry point — iterates subjects, calls crop then feature builder |
| `preprocess_raw.py` | Top-level crop function — loads motion + label files and finds the overlapping interval |
| `raw_data_processor.py` | Coordinates cropping and epoch validation across all subjects |
| `feature_builder.py` | Builds FFT features for a single subject; saves `.out` files; contains the skip-check |
| `epoch.py` | `Epoch` dataclass: timestamp + 30-second window |
| `interval.py` | `Interval` dataclass: start/end timestamps; used for cropping |
| `time_service.py` | Utility: converts timestamps and computes epoch grids |

## Sub-packages

| Package | Purpose |
|---|---|
| `motion/` | Loads raw accelerometer CSVs into `MotionCollection` objects |
| `motion_fft/` | Computes FFT (1–30 Hz) on each 30-second epoch for x, y, z, and vector magnitude |
| `psg/` | Loads sleep-stage label CSVs; handles cropping and epoch-level timestamp saving |
| `heart_rate/` | Legacy code retained for import compatibility — not called in this fork |

## Feature extraction detail

For each 30-second epoch, four FFT feature arrays are computed:

| Feature type | Enum name | Shape |
|---|---|---|
| x-axis FFT bins 1–30 | `motion_xfft1_30` | (30,) |
| y-axis FFT bins 1–30 | `motion_yfft1_30` | (30,) |
| z-axis FFT bins 1–30 | `motion_zfft1_30` | (30,) |
| Vector magnitude FFT 1–30 | `motion_vmfft1_30` | (30,) |

Each feature array is saved as `<subject_id>_<feature_name>.out` inside `data/data_processed/agv/features/`.

Two additional files are saved per subject:
- `<subject_id>_psg_labels.out` — epoch-level binary labels (0=Wake, 1=Sleep)
- `<subject_id>_psg_timestamps.out` — epoch-level Unix timestamps (seconds)

## Skip logic

`feature_builder.py` checks whether the x-axis FFT file already exists before processing a subject.
If it does, that subject is skipped. This means you can safely re-run `preprocessing_feature.py`
after a partial failure without re-computing features from scratch.

To force re-extraction for a subject, delete their `.out` files in `features/` and re-run.
