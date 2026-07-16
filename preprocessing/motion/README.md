# preprocessing/motion/

Loads raw accelerometer CSV files and exposes them as typed collection objects for use by the feature extraction pipeline.

## Files

| File | Purpose |
|---|---|
| `motion_service.py` | Reads `<subject_id>_motion.csv` and returns a `MotionCollection` |
| `motion_collection.py` | `MotionCollection` dataclass — holds x, y, z, magnitude, ENMO arrays and timestamps |
| `motion_vm_collection.py` | Variant that stores only the vector magnitude channel |

## Motion CSV format

Written by `data_ingestion.py`:

| Column | Description |
|---|---|
| `agvTime` | Relative time (seconds from start of recording) |
| `agvx` | Raw x-axis acceleration (g) |
| `agvy` | Raw y-axis acceleration (g) |
| `agvz` | Raw z-axis acceleration (g) |
| `agvmagnitude` | Vector magnitude = √(x²+y²+z²) |
| `agvenmo` | ENMO = max(magnitude − 1, 0) |
| `timestamp` | Absolute Unix epoch (seconds) |

## Notes

ENMO (Euclidean Norm Minus One) is a standard actigraphy activity measure used in wearable research.
It is available in the motion file for potential downstream use, but the LSTM model itself uses the
raw x/y/z/magnitude FFT features rather than ENMO directly.
