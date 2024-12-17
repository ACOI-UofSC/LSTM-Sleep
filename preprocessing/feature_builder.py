from source.constants import Constants
from preprocessing.motion_fft.motion_fft_feature_service import MotionFFTFeatureService
from preprocessing.heart_rate.heart_rate_feature_service import HeartRateFeatureService
from preprocessing.psg.psg_label_service import PSGLabelService
from preprocessing.raw_data_processor import RawDataProcessor

feature_name = {
    'biobank': ['enmoTrunc', 'enmoAbs', 'xMean', 'yMean', 'zMean', 'xRange', 'yRange', 'zRange', 'xStd',
                'yStd', 'zStd', 'xyCov', 'xzCov', 'yzCov', 'entropy', 'MPD', 'skew', 'kurt', 'avgArmAngel',
                'avgArmAngelAbsDiff', 'f1', 'p1', 'f2', 'p2', 'f625', 'p625', 'totalPower'],
    'tlbc': ["fMean", "fStd", "fCoefVariation", "fMedian", "fMin", "fMax", "f25thP", "f75thP", "fAutocorr",
             "fCorrxy", "fCorrxz", "fCorryz", "fAvgRoll", "fAvgPitch", "fAvgYaw", "fSdRoll", "fSdPitch", "fSdYaw",
             "fRollG", "fPitchG", "fYawG", "fFmax", "fPmax", "fFmaxBand", "fPmaxBand", "fEntropy", "vMFFT0", "FFT1",
             "vFFT2", "vFFT3", "vFFT4", "vFFT5", "vFFT6", "vFFT7", "vFFT8", "vFFT9", "vFFT10", "vFFT11", "vFFT12",
             "vFFT13", "vFFT14"],
    'ggir': ['BFEN', 'LFEN', 'LFENMO', 'HFEN', 'HFENplus', 'roll_med_acc_x', 'roll_med_acc_y', 'roll_med_acc_z',
             'dev_roll_med_acc_x', 'dev_roll_med_acc_y', 'dev_roll_med_acc_z', 'angle_x', 'angle_y', 'angle_z',
             'ENMO', 'MAD', 'EN', 'ENMOa']

}


class FeatureBuilder(object):

    @staticmethod
    def build(subject_id):
        if Constants.VERBOSE:
            print("Getting valid epochs...")
        valid_epochs = RawDataProcessor.get_valid_epochs(subject_id)[:-1]

        if Constants.VERBOSE:
            print("Building features...")
        FeatureBuilder.build_labels(subject_id, valid_epochs)
        FeatureBuilder.build_from_wearables(subject_id, valid_epochs)

    @staticmethod
    def build_labels(subject_id, valid_epochs):
        psg_labels = PSGLabelService.build(subject_id, valid_epochs)
        PSGLabelService.write(subject_id, psg_labels)

    @staticmethod
    def build_from_wearables(subject_id, valid_epochs):
        heart_rate_feature = HeartRateFeatureService.build(subject_id, valid_epochs, 30)
        motion_xfft30_feature = MotionFFTFeatureService.build_direction_fft(subject_id, valid_epochs, 'x', '1-30')
        motion_yfft30_feature = MotionFFTFeatureService.build_direction_fft(subject_id, valid_epochs, 'y', '1-30')
        motion_zfft30_feature = MotionFFTFeatureService.build_direction_fft(subject_id, valid_epochs, 'z', '1-30')
        motion_vmfft30_feature = MotionFFTFeatureService.build_vmfft(subject_id, valid_epochs, '1-30')

        MotionFFTFeatureService.write_dir_fft(subject_id, motion_xfft30_feature, 'x', '1-30')
        MotionFFTFeatureService.write_dir_fft(subject_id, motion_yfft30_feature, 'y', '1-30')
        MotionFFTFeatureService.write_dir_fft(subject_id, motion_zfft30_feature, 'z', '1-30')
        MotionFFTFeatureService.write_vmfft(subject_id, motion_vmfft30_feature, '1-30')
        HeartRateFeatureService.write(subject_id, heart_rate_feature, 30)
        
