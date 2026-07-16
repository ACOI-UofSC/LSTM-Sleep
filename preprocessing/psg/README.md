# preprocessing/psg/

Handles polysomnography (PSG) label loading, time-window cropping, and epoch-level data preparation. In this actigraphy-only fork, "PSG" refers to the ground-truth sleep/wake labels — they do not need to come from a clinical PSG system; any binary label file will work.

## Files

| File | Purpose |
|---|---|
| `psg_service.py` | Loads `<subject_id>_labeled.csv` into a time-indexed array |
| `psg_label_service.py` | Builds, saves, and loads epoch-level label and timestamp `.out` files |
| `psg_raw_data_collection.py` | Holds raw label data before epoch extraction |
| `stage_item.py` | Single stage entry: timestamp + `SleepStage` enum value |
| `psg_converter.py` | Converts string stage labels to `SleepStage` enum values |
| `psg_file_type.py` | Enum for PSG file format variants (legacy, kept for compatibility) |
| `compumedics_processor.py` | Parser for Compumedics EDF export format (legacy) |
| `vitaport_processor.py` | Parser for Vitaport EDF export format (legacy) |
| `psg_report_processor.py` | Parser for text-based PSG report format (legacy) |
| `report_summary.py` | Data container for parsed PSG report metadata (legacy) |

## Label file format

Written by `data_ingestion.py`:

| Column | Description |
|---|---|
| `psgtime` | Original timestamp string |
| `psgstg` | Stage string: `"W"` (Wake) or `"N2"` (Sleep proxy) |
| `labels` | Numeric label: `0` (Wake) or `1` (Sleep) |
| `timestamp` | Unix epoch (seconds) |
| `Time` | Relative time from recording start |

## Key additions in this fork

`psg_label_service.py` was extended with two new methods:

- `build_timestamps(valid_epochs)` — extracts the Unix timestamp for each valid 30-second epoch
- `write_timestamps(subject_id, timestamps)` — saves them to `features/<subject_id>_psg_timestamps.out`

These timestamps flow through to the final prediction CSVs so each output row can be matched to
an absolute clock time, enabling downstream time-series analysis and comparison with GGIR output.

## Label encoding

> **0 = Wake, 1 = Sleep** — this is enforced at the enum level (`SleepWakeLabel`) and verified by unit tests in `test_agv_pipeline.py`.
