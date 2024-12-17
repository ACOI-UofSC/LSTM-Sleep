class RawPerformance(object):
    def __init__(self, true_labels, class_probabilities, subject=None, predicted_labels=None, feats=None):
        self.true_labels = true_labels
        self.class_probabilities = class_probabilities
        self.subject = subject
        self.predicted_labels = predicted_labels
        self.feature_importance = feats

