import numpy as np

from source.analysis.setup.sleep_labeler import SleepLabeler


class ClassifierInputBuilder(object):

    @staticmethod
    def get_array(subject_ids, subject_dictionary, feature_set):

        all_subjects_features = []
        all_subjects_labels = []

        for subject_id in subject_ids:
            subject_features = []
            subject = subject_dictionary[subject_id]
            feature_dictionary = subject.feature_dictionary

            for feature in feature_set:
                feature_data = feature_dictionary[feature]
                if len(np.where(np.isnan(feature_data))[0]) > 0:
                    # print(subject_id, feature)
                    feature_data[np.isnan(feature_data)] = 0
                feature_data = np.expand_dims(feature_data, axis=-1)
                subject_features.append(feature_data)

            subject_features = np.concatenate(subject_features, axis=-1)
            subject_labels = subject.labeled_sleep.reshape(-1)

            all_subjects_features.append(subject_features)
            all_subjects_labels.append(subject_labels)

        return all_subjects_features, all_subjects_labels

    @staticmethod
    def get_sleep_wake_inputs(subject_ids, subject_dictionary, feature_set):
        values, raw_labels = ClassifierInputBuilder.get_array(subject_ids, subject_dictionary, feature_set)
        processed_labels = SleepLabeler.label_sleep_wake(raw_labels)
        return values, processed_labels


    @staticmethod
    def get_four_class_inputs(subject_ids, subject_dictionary, feature_set):
        values, raw_labels = ClassifierInputBuilder.get_array(subject_ids, subject_dictionary, feature_set)
        processed_labels = SleepLabeler.label_four_class(raw_labels)
        return values, processed_labels


    @staticmethod
    def __append_feature(array, feature):
        if len(np.shape(feature)) < 2:
            feature = np.transpose([feature])
        if np.shape(array)[0] == 0:
            array = feature
        else:
            array = np.hstack((array, feature))

        return array

    @staticmethod
    def __stack(combined_array, new_array):
        if np.shape(combined_array)[0] == 0:
            combined_array = new_array
        else:
            combined_array = np.vstack((combined_array, new_array))
        return combined_array
