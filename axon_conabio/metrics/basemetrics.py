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

    def __init__(self):
        self._validate()

    def _validate(self):
        if (not isinstance(self.name, str) or len(self.name) == 0):
            raise ValueError('Please name this metric.')

    def label_preprocess(self, label):
        return label

    def prediction_preprocess(self, prediction):
        return prediction

    def evaluation_postprocess(self, results):
        return {self.name: results}

    @abstractmethod
    def evaluate(self, prediction, label):
        pass

    def __call__(
            self,
            prediction,
            label,
            with_prediction_preprocess=True,
            with_label_preprocess=True,
            with_evaluation_postprocess=True):
        if with_prediction_preprocess:
            prediction = self.prediction_preprocess(prediction)

        if with_label_preprocess:
            label = self.label_preprocess(label)

        evaluation = self.evaluate(prediction, label)

        if with_evaluation_postprocess:
            evaluation = self.evaluation_postprocess(evaluation)

        return evaluation


@six.add_metaclass(ABCMeta)
class MetricBundle(Metric):
    @property
    @abstractmethod
    def names(self):
        pass

    def _validate(self):
        if not isinstance(self.names, (list, tuple)):
            raise ValueError('Metric bundle names not defined.')

    def evaluation_postprocess(self, results):
        dictionary_results = {
            name: result
            for name, result in zip(self.names, results)}
        return dictionary_results


@six.add_metaclass(ABCMeta)
class ThresholdedMetric(Metric):
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __call__(self, prediction, label):
        results = defaultdict(dict)

        proc_prediction = self.prediction_preprocess(prediction)
        proc_label = self.label_preprocess(label)

        for threshold in self.thresholds:
            thresholded_prediction = proc_prediction >= threshold
            evaluation = super(ThresholdedMetric, self).__call__(
                thresholded_prediction,
                proc_label,
                with_label_preprocess=False,
                with_prediction_preprocess=False)

            for key in evaluation:
                results[key][threshold] = evaluation[key]

        return dict(results)
