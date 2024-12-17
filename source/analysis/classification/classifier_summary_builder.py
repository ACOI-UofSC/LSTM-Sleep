from source.analysis.classification.classifier_service import ClassifierService
from source.analysis.classification.classifier_summary import ClassifierSummary
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.data_split import DataSplit
from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.train_test_splitter import TrainTestSplitter


class SleepWakeClassifierSummaryBuilder(object):
    @staticmethod
    def run_feature_sets(data_splits: [DataSplit], subject_dictionary, attributed_classifier: AttributedClassifier,
                         feature_sets: [[FeatureType]]):
        performance_dictionary = {}
        for feature_set in feature_sets:
            raw_performance_results = ClassifierService.run_sw(data_splits, attributed_classifier,
                                                               subject_dictionary, feature_set)
            performance_dictionary[tuple(feature_set)] = raw_performance_results

        return ClassifierSummary(attributed_classifier, performance_dictionary)

    @staticmethod
    def build_leave_multiple_out(attributed_classifier: AttributedClassifier,
                                 feature_sets: [[FeatureType]], kfold: int) -> ClassifierSummary:
        subject_dictionary = SubjectBuilder.get_subject_dictionary()

        data_splits = TrainTestSplitter.by_number_age(subject_dictionary, kfold)

        return SleepWakeClassifierSummaryBuilder.run_feature_sets(data_splits, subject_dictionary,
                                                                  attributed_classifier,
                                                                  feature_sets)


class MultiClassClassifierSummaryBuilder(object):

    @staticmethod
    def run_feature_sets(data_splits: [DataSplit], subject_dictionary, attributed_classifier: AttributedClassifier,
                         feature_sets: [[FeatureType]]):
        performance_dictionary = {}
        for feature_set in feature_sets:
            raw_performance_results = ClassifierService.run_four_class(data_splits, attributed_classifier,
                                                                       subject_dictionary, feature_set)
            performance_dictionary[tuple(feature_set)] = raw_performance_results

        return ClassifierSummary(attributed_classifier, performance_dictionary)

    @staticmethod
    def build_leave_multiple_out(attributed_classifier: AttributedClassifier,
                                 feature_sets: [[FeatureType]], kfold: int) -> ClassifierSummary:
        subject_dictionary = SubjectBuilder.get_subject_dictionary()

        data_splits = TrainTestSplitter.by_number_age(subject_dictionary, kfold)

        return MultiClassClassifierSummaryBuilder.run_feature_sets(data_splits, subject_dictionary,
                                                                   attributed_classifier,
                                                                   feature_sets)
