import time

import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

from source.analysis.classification.classifier_input_builder import ClassifierInputBuilder
from source.analysis.dataset import TimeSeriesDataset
from source.analysis.performance.raw_performance import RawPerformance
from source.constants import Constants

import torch

from source.analysis.model import Trainer


class ClassifierService(object):

    @staticmethod
    def run_sw(data_splits, classifier, subject_dictionary, feature_set):
        return ClassifierService.run_in_parallel(ClassifierService.run_single_data_split_sw,
                                                 data_splits, classifier,
                                                 subject_dictionary, feature_set)

    @staticmethod
    def run_four_class(data_splits, classifier, subject_dictionary, feature_set):
        return ClassifierService.run_in_parallel(ClassifierService.run_single_data_split_four_class,
                                                 data_splits, classifier,
                                                 subject_dictionary, feature_set)

    @staticmethod
    def run_bin_overall(classifier, subject_dictionary, feature_set, test_frac):
        x, y, ids = ClassifierInputBuilder.get_sleep_wake_inputs(subject_ids=list(subject_dictionary.keys()),
                                                                 subject_dictionary=subject_dictionary,
                                                                 feature_set=feature_set)
        X = np.concatenate([x, ids.reshape(-1, 1)], axis=1)
        X_train, X_test, training_y, testing_y = train_test_split(X, y, test_size=test_frac, stratify=y,
                                                                  random_state=42)
        training_x, training_y_ids = np.float64(X_train[:, :-1]), X_train[:, -1]
        testing_x, testing_y_ids = np.float64(X_test[:, :-1]), X_test[:, -1]
        return [ClassifierService.run_single_data_split(training_x, training_y, testing_x, testing_y,
                                                        classifier, testing_ids=testing_y_ids)]

    @staticmethod
    def run_in_parallel(function, data_splits, classifier, subject_dictionary, feature_set):
        results = [function(i, classifier, subject_dictionary, feature_set) for i in data_splits]
        return results

    @staticmethod
    def run_single_data_split_sw(data_split, attributed_classifier, subject_dictionary, feature_set):

        training_x, training_y = ClassifierInputBuilder.get_sleep_wake_inputs(data_split.training_set,
                                                                              subject_dictionary=subject_dictionary,
                                                                              feature_set=feature_set)
        testing_x, testing_y = ClassifierInputBuilder.get_sleep_wake_inputs(data_split.testing_set,
                                                                            subject_dictionary=subject_dictionary,
                                                                            feature_set=feature_set)

        return ClassifierService.run_single_data_split(training_x, training_y, testing_x, testing_y,
                                                       attributed_classifier, testing_ids=data_split.testing_set)

    @staticmethod
    def run_single_data_split_four_class(data_split, attributed_classifier, subject_dictionary, feature_set):

        training_x, training_y = ClassifierInputBuilder.get_four_class_inputs(data_split.training_set,
                                                                              subject_dictionary=subject_dictionary,
                                                                              feature_set=feature_set)

        testing_x, testing_y = ClassifierInputBuilder.get_four_class_inputs(data_split.testing_set,
                                                                            subject_dictionary=subject_dictionary,
                                                                            feature_set=feature_set)

        return ClassifierService.run_single_data_split(training_x, training_y, testing_x, testing_y,
                                                       attributed_classifier, testing_ids=data_split.testing_set)

    @staticmethod
    def run_single_data_split(training_x, training_y, testing_x, testing_y, attributed_classifier, testing_ids=None):
        start_time = time.time()

        # 计算类别权重
        all_labels = np.concatenate(training_y)
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(all_labels),
                                                          y=all_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float)

        # 数据加载器
        dataset = TimeSeriesDataset(training_x, training_y)
        # 数据集划分
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        test_dataset = TimeSeriesDataset(testing_x, testing_y)

        trainer = Trainer(attributed_classifier.classifier, class_weight=class_weights, device='cpu')
        trainer.set_train_data(train_dataset)
        trainer.set_val_data(val_dataset)
        trainer.set_test_data(test_dataset)
        trainer.fit()
        class_probabilities, predicted_labels, testing_y = trainer.test()

        raw_performance = RawPerformance(true_labels=testing_y, class_probabilities=class_probabilities,
                                         subject=testing_ids, predicted_labels=predicted_labels)

        if Constants.VERBOSE:
            print('Completed data split in ' + str(time.time() - start_time))

        return raw_performance

