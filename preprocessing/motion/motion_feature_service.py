import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from preprocessing.epoch import Epoch
from preprocessing.motion.motion_service import MotionService


class MotionFeatureService(object):
    WINDOW_SIZE = 10 * 30 - 15

    @staticmethod
    def load(subject_id):
        motion_feature_path = MotionFeatureService.get_path(subject_id)
        feature = pd.read_csv(str(motion_feature_path)).values
        return feature

    @staticmethod
    def load_mpd(subject_id):
        motion_feature_path = MotionFeatureService.get_mpd_path(subject_id)
        feature = pd.read_csv(str(motion_feature_path)).values
        return feature


    @staticmethod
    def get_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_motion_feature.out')

    @staticmethod
    def get_mpd_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_motion_mpd_feature.out')

    @staticmethod
    def write(subject_id, feature):
        motion_feature_path = MotionFeatureService.get_path(subject_id)
        np.savetxt(motion_feature_path, feature, fmt='%f')

    @staticmethod
    def write_mpd(subject_id, feature):
        motion_feature_path = MotionFeatureService.get_mpd_path(subject_id)
        np.savetxt(motion_feature_path, feature, fmt='%f')

    @staticmethod
    def build_mpd(subject_id, valid_epochs):
        motion_vm_conllection = MotionService.load_cropped(subject_id)
        return MotionFeatureService.build_mpd_from_vm_collection(motion_vm_conllection, valid_epochs)

    @staticmethod
    def build_mpd_from_vm_collection(motion_vm_conllection, valid_epochs):
        motion_mpd_features = []

        interpolated_timestamps, interpolated_mpd = MotionFeatureService.interpolate(
            motion_vm_conllection)

        for epoch in valid_epochs:
            indices_in_range = MotionFeatureService.get_window(interpolated_timestamps, epoch)
            motion_mpd_values_in_range = interpolated_mpd[indices_in_range]
            mean_mpd_in_range = np.mean(motion_mpd_values_in_range)
            feature = np.sqrt((motion_mpd_values_in_range - mean_mpd_in_range) ** 2 / len(motion_mpd_values_in_range))
            feature = MotionFeatureService.get_feature(feature)
            motion_mpd_features.append(feature)

        return np.array(motion_mpd_features)

    @staticmethod
    def interpolate(motion_vm_conllection):
        timestamps = motion_vm_conllection.timestamps.flatten()
        motion_vm_values = motion_vm_conllection.values[:,-1].flatten()
        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        interpolated_vm = np.interp(interpolated_timestamps, timestamps, motion_vm_values)

        return interpolated_timestamps, interpolated_vm

    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - MotionFeatureService.WINDOW_SIZE
        end_time = epoch.timestamp + Epoch.DURATION + MotionFeatureService.WINDOW_SIZE
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]

    @staticmethod
    def get_feature(motion_mpd_values):
        convolution = utils.smooth_gauss(motion_mpd_values.flatten(), np.shape(motion_mpd_values.flatten())[0])
        return np.array([convolution])