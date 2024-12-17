from pathlib import Path

class Constants(object):

    WAKE_THRESHOLD = 0.5  #
    REM_THRESHOLD = 0.35

    INCLUDE_CIRCADIAN = False
    EPOCH_DURATION_IN_SECONDS = 30
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_DAY = 3600 * 24
    SECONDS_PER_HOUR = 3600
    VERBOSE = True
    # DEVICE = 'actigraph'
    # ECG = 'apple'

    DEVICE = 'apple'

    INPUT_ROOT = Path('../data/data_processed/')

    PSG_FILE_PATH = Path(f'{INPUT_ROOT}/{DEVICE}/labels/')
    CROPPED_FILE_PATH = Path(f'{INPUT_ROOT}/{DEVICE}/cropped/')
    FEATURE_FILE_PATH = Path(f'{INPUT_ROOT}/{DEVICE}/features/')
    MOTION_FILE_PATH = Path(f'{INPUT_ROOT}/{DEVICE}/motion/')
    HR_FILE_PATH = Path(f'{INPUT_ROOT}/{DEVICE}/heart_rate/')

    if not CROPPED_FILE_PATH.exists():
        CROPPED_FILE_PATH.mkdir(parents=True, exist_ok=True)
    if not FEATURE_FILE_PATH.exists():
        FEATURE_FILE_PATH.mkdir(parents=True, exist_ok=True)

    AGE_FILE_PATH = Path(f'{INPUT_ROOT}/subjects_age.csv')
    DIAGNOSES = Path(f'{INPUT_ROOT}/sleep_diagnoses.csv')
    LOWER_BOUND = -0.2
