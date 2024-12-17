import time
import sys

sys.path.append('..')
from source.analysis.setup.subject_builder import SubjectBuilder
from source.constants import Constants
from preprocessing.feature_builder import FeatureBuilder
from preprocessing.raw_data_processor import RawDataProcessor
from joblib import Parallel, delayed


def run_preprocessing(subject_set):
    start_time = time.time()
    parallel = Parallel(n_jobs=-1)
    parallel(delayed(RawDataProcessor.crop_all)(str(subject_set[i])) for i in (range(len(subject_set))))

    parallel(delayed(FeatureBuilder.build)(str(subject_set[i])) for i in (range(len(subject_set))))

    end_time = time.time()
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")


if __name__ == '__main__':
    subject_ids = SubjectBuilder.get_all_subject_ids_from_path(Constants.PSG_FILE_PATH)

    run_preprocessing(subject_ids)
