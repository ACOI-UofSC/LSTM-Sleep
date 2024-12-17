import os
import time
import sys

sys.path.append("/")
sys.path.append("..")
sys.path.append("../..")

from source.analysis.model import LocalGlobalLSTM
from source.analysis.setup.attributed_classifier import AttributedClassifier

from source import utils
from source.analysis.classification.classifier_summary_builder import SleepWakeClassifierSummaryBuilder, \
    MultiClassClassifierSummaryBuilder

from source.analysis.classification.classifier_summary import ClassifierSummary
from source.constants import Constants

import torch
import random
import numpy as np
import pandas as pd


def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def save_predictions_binary(classifier_summary: ClassifierSummary, path_mark='1207'):
    for feature_set in classifier_summary.performance_dictionary:
        path = f'./{path_mark}/{classifier_summary.attributed_classifier.name}/{"+".join([i.value for i in feature_set])}/'
        if not os.path.exists(path):
            os.makedirs(path)

        results = {'subject': [], 'epoch': [], 'reference': [], 'device': []}
        score = []
        raw_performances = classifier_summary.performance_dictionary[feature_set]
        for p in range(len(raw_performances)):
            performance = raw_performances[p]
            for i in range(len(performance.true_labels)):
                ref = list(performance.true_labels[i])
                device = list(performance.predicted_labels[i])
                cur_subids = [performance.subject[i]] * len(device)
                epoch = list(range(1, len(device) + 1))

                results['subject'] = results['subject'] + cur_subids
                results['epoch'] = results['epoch'] + epoch
                results['reference'] = results['reference'] + ref
                results['device'] = results['device'] + device
                score.append(performance.class_probabilities[i])
        score = np.concatenate(score)
        score = pd.DataFrame(score, columns=[f'score_{i}' for i in range(score.shape[1])])
        results_pd = pd.DataFrame(results)

        results_pd = pd.concat([results_pd, score], axis=1)
        results_pd.to_csv(
            f'./{path_mark}/{classifier_summary.attributed_classifier.name}/{"+".join([i.value for i in feature_set])}/class_probabilities.csv',
            index=False)


def train(feature_sets, class_num=4):
    print(f'{class_num} class')

    classifiers = AttributedClassifier(name='LocalGlobalLSTM',
                                       classifier=LocalGlobalLSTM(feature_dim=len(feature_sets[0]), local_steps=30,
                                                                  n_class=class_num))
    print('Running ' + classifiers.name + '...')

    if class_num == 2:
        classifier_summary = SleepWakeClassifierSummaryBuilder.build_leave_multiple_out(classifiers,
                                                                                        feature_sets, 10)
    elif class_num == 4:
        classifier_summary = MultiClassClassifierSummaryBuilder.build_leave_multiple_out(classifiers,
                                                                                         feature_sets, 10)
    save_predictions_binary(classifier_summary,
                            f'{Constants.DEVICE}_4class/{Constants.DEVICE}_lstm_xyzvmhr_fft30')


if __name__ == "__main__":
    set_random_seed(42)
    start_time = time.time()
    print(Constants.DEVICE)
    feature_sets = utils.get_lstm_feature_fft30_sets()

    train(feature_sets)
    end_time = time.time()

    print('Elapsed time to generate figure: ' + str((end_time - start_time) / 60) + ' minutes')
