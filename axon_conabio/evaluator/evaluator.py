import os
import sys
import logging

import six
from tqdm import tqdm
import tensorflow as tf

from ..datasets.basedataset import Dataset
from ..models.basemodel import Model
from ..metrics.basemetrics import Metric
from ..utils import TF_DTYPES


class Evaluator(object):

    def __init__(
            self,
            path,
            checkpoints_dir='checkpoints',
            evaluations_dir='evaluations'):

        self.path = path
        self.checkpoints_dir = os.path.join(path, checkpoints_dir)
        self.evaluations_dir = os.path.join(path, evaluations_dir)

        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

        if not os.path.exists(self.evaluations_dir):
            os.makedirs(self.evaluations_dir)

    def _load_latest_checkpoint(self):
        ckpt = tf.train.latest_checkpoint(self.checkpoints_dir)
        if ckpt is None:
            msg = 'No model checkpoint was found at {path}.\n'
            msg = 'Exiting evaluation.'
            msg = msg.format(path=self.checkpoints_dir)
            logging.error(msg)
            sys.exit()
            quit()

        return ckpt

    def _build_inputs(self, dataset):
        input_structure = dataset.input_structure

        def get_dtype(args):
            if len(args) == 1:
                return tf.float32
            else:
                dtype_string = args[1]
                return TF_DTYPES[dtype_string]

        inputs = {
            key: tf.placeholder(get_dtype(value), shape=([1] + value[0]))
            for key, value in six.iteritems(input_structure)
        }

        if len(inputs) == 1:
            _, inputs = inputs.popitem()

        return inputs

    def _make_feed_dict(self, input_tensors, inputs):
        feed_dict = {}

        # Case for structured inputs
        if isinstance(input_tensors, dict):
            for key, value in six.iteritems(input_tensors):
                try:
                    input_value = inputs[key]
                except KeyError:
                    msg = 'Declared dataset input structures does not '
                    msg += 'coincided with test iterator outputs.'
                    raise KeyError(msg)
                feed_dict[value] = [input_value]

        # Case for single input
        else:
            feed_dict[input_tensors] = [inputs]

        return feed_dict

    def evaluate(self, model=None, dataset=None, metrics=None):
        # Check for correct types
        assert issubclass(model, Model)
        assert issubclass(dataset, Dataset)
        assert isinstance(metrics, (list, tuple))
        for metric in metrics:
            assert issubclass(metric, Metric)

        # Check if model checkpoint exists
        ckpt = self._load_latest_checkpoint()

        # Instantiate metrics
        metrics = [metric() for metric in metrics]

        # Create new graph for model evaluation
        graph = tf.Graph()

        # Instantiate model with graph
        model_instance = model(graph=graph)

        # Create input pipeline
        with graph.as_default():
            dataset_instance = dataset()
            input_tensors = self._build_inputs(dataset)

        prediction_tensor = model_instance.predict(input_tensors)

        sess = tf.Session(graph=graph)
        model_instance.restore(sess, ckpt)

        evaluations = []
        for id_, inputs, label in tqdm(dataset_instance.iter_test()):
            feed_dict = self._make_feed_dict(input_tensors, inputs)
            prediction = sess.run(prediction_tensor, feed_dict=feed_dict)

            results = {'id': id_}
            for metric in metrics:
                results.update(metric(prediction, label))

            evaluations.append(results)

        return evaluations
