from __future__ import division

from abc import abstractmethod, ABCMeta
from collections import defaultdict
import six


@six.add_metaclass(ABCMeta)
class Metric(object):
    @property
    @abstractmethod
    def name(self):
        pass

    def label_preprocess(self, label):
        return label

    def prediction_preprocess(self, prediction):
        return prediction

    def evaluation_postprocess(self, results):
        return {self.name: results}

    @abstractmethod
    def evaluate(self, prediction, label):
        pass

    def __call__(self, prediction, label):
        prediction = self.prediction_preprocess(prediction)
        label = self.label_preprocess(label)
        evaluation = self.evaluate(prediction, label)
        evaluation = self.evaluation_postprocess(evaluation)
        return evaluation


@six.add_metaclass(ABCMeta)
class MetricBundle(Metric):
    @property
    @abstractmethod
    def names(self):
        pass

    def evaluation_postprocess(self, results):
        dictionary_results = {
            name: result
            for name, result in zip(self.names, results)}
        return dictionary_results


@six.add_metaclass(ABCMeta)
class ThresholdedMetric(Metric):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    @abstractmethod
    def process_threshold(self, prediction, threshold):
        pass

    @staticmethod
    def get_threshold_key(key, threshold):
        return '{}_@_{}'.format(key, threshold)

    def __call__(self, prediction, label):
        results = defaultdict(dict)

        proc_prediction = self.prediction_preprocess(prediction)
        proc_label = self.label_preprocess(label)

        for threshold in self.thresholds:
            thresholded_prediction = self.process_threshold(
                proc_prediction, threshold)

            evaluation = self.evaluate(thresholded_prediction, proc_label)
            evaluation = self.evaluation_postprocess(evaluation)

            for key in evaluation:
                result_key = self.get_threshold_key(key, threshold)
                results[result_key] = evaluation[key]

        return dict(results)
