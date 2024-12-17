import seaborn as sns

from source.analysis.setup.feature_type import FeatureType


class FeatureSetService(object):

    @staticmethod
    def get_label(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.count}:
            return 'Motion only'
        if set(feature_set) == {FeatureType.heart_rate}:
            return 'HR only'
        if set(feature_set) == {FeatureType.motion_vmfft9}:
            return 'VMFFT9 only'
        if set(feature_set) == {FeatureType.motion_vmfft14}:
            return 'VMFFT14 only'
        if set(feature_set) == {FeatureType.motion_xfft4}:
            return 'xFFT4 only'
        if set(feature_set) == {FeatureType.motion_xfft9}:
            return 'xFFT9 only'
        if set(feature_set) == {FeatureType.motion_mpd}:
            return 'MPD only'
        if set(feature_set) == {FeatureType.bfen}:
            return 'BFEN only'
        if set(feature_set) == {FeatureType.angley}:
            return 'y-Angle only'
        if set(feature_set) == {FeatureType.y_offset_angle}:
            return 'y-Offset Angle only'
        if set(feature_set) == {FeatureType.motion_vmfft9, FeatureType.motion_mpd}:
            return 'VMFFT9, MPD'
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate}:
            return 'Motion, HR'
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model}:
            return 'Motion, HR, and Clock'
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.cosine}:
            return 'Motion, HR, and Cosine'
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.time}:
            return 'Motion, HR, and Time'
        if set(feature_set) == {FeatureType.age, FeatureType.y_offset_angle, FeatureType.motion_xfft4,
                                FeatureType.motion_vmfft9,
                                FeatureType.motion_mpd,
                                FeatureType.bfen, FeatureType.motion_vmfft14, FeatureType.motion_xfft9,
                                FeatureType.angley,
                                FeatureType.pmaxband, FeatureType.count, FeatureType.heart_rate, FeatureType.cosine}:
            return 'All Features'

    @staticmethod
    def get_color(feature_set: [FeatureType]):
        if set(feature_set) == {FeatureType.count}:
            return sns.xkcd_rgb["denim blue"]
        if set(feature_set) == {FeatureType.heart_rate}:
            return sns.xkcd_rgb["yellow orange"]
        if set(feature_set) == {FeatureType.heart_rate}:
            return sns.xkcd_rgb["red"]
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate}:
            return sns.xkcd_rgb["medium green"]
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.circadian_model}:
            return sns.xkcd_rgb["medium pink"]
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.cosine}:
            return sns.xkcd_rgb["plum"]
        if set(feature_set) == {FeatureType.count, FeatureType.heart_rate, FeatureType.time}:
            return sns.xkcd_rgb["greyish"]
        if set(feature_set) == {FeatureType.age, FeatureType.y_offset_angle, FeatureType.motion_xfft4,
                                FeatureType.motion_vmfft9,
                                FeatureType.motion_mpd,
                                FeatureType.bfen, FeatureType.motion_vmfft14, FeatureType.motion_xfft9,
                                FeatureType.angley,
                                FeatureType.pmaxband, FeatureType.count, FeatureType.heart_rate, FeatureType.cosine}:
            return sns.xkcd_rgb["orangered"]
