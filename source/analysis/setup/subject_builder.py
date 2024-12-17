import pandas as pd
import numpy as np
import os
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject import Subject
from source.constants import Constants
from preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from preprocessing.motion_fft.motion_fft_feature_service import MotionFFTFeatureService
from preprocessing.psg.psg_label_service import PSGLabelService


class SubjectBuilder(object):
    @staticmethod
    def get_all_subject_ids_from_path(dir_path):
        root, dirs, files = next(os.walk(dir_path))
        subjects_as_ints = [i.split('_')[0] if '_' in i else i.split('.')[0] for i in files if '._' not in i]
        subjects_as_ints = sorted(list(set(subjects_as_ints)))
        return subjects_as_ints

    @staticmethod
    def get_subject_dictionary():
        subject_dictionary = {}
        all_subject_ids = SubjectBuilder.get_all_subject_ids_from_path(Constants.FEATURE_FILE_PATH)
        # ecg_subject_ids = SubjectBuilder.get_all_subject_ids_from_path(Constants.ECG_FILE_PATH)
        # all_subject_ids = [i for i in all_subject_ids if i in ecg_subject_ids]

        print(f'input subjects: {len(all_subject_ids)}')
        for subject_id in all_subject_ids:
            cur_sub = SubjectBuilder.build(subject_id)
            subject_dictionary[subject_id] = cur_sub
        print(f'>200 samples subjects: {len(subject_dictionary.keys())}')
        return subject_dictionary

    @staticmethod
    def build(subject_id):

        feature_motion_xfft130 = MotionFFTFeatureService.load_dir_fft(subject_id, 'x', '1-30')
        feature_motion_yfft130 = MotionFFTFeatureService.load_dir_fft(subject_id, 'y', '1-30')
        feature_motion_zfft130 = MotionFFTFeatureService.load_dir_fft(subject_id, 'z', '1-30')
        feature_motion_vmfft130 = MotionFFTFeatureService.load_vmfft(subject_id, '1-30')

        feature_hr = HeartRateFeatureService.load(subject_id, 30)
        labeled_sleep = PSGLabelService.load(subject_id)

        subject_ages = pd.read_csv(Constants.AGE_FILE_PATH)
        feature_ages = subject_ages[subject_ages['Child ID'] == int(subject_id)].iloc[0, 2]
        feature_ages = np.array([feature_ages] * len(feature_hr)).reshape(-1, 1)

        feature_dictionary = {
            FeatureType.motion_xfft1_30: feature_motion_xfft130,
            FeatureType.motion_yfft1_30: feature_motion_yfft130,
            FeatureType.motion_zfft1_30: feature_motion_zfft130,
            FeatureType.motion_vmfft1_30: feature_motion_vmfft130,
            FeatureType.age: feature_ages,
            FeatureType.heart_rate_fft1_30: feature_hr
        }

        subject = Subject(subject_id=subject_id,
                          labeled_sleep=labeled_sleep,
                          feature_dictionary=feature_dictionary)

        return subject
