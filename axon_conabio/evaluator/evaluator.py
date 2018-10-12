import os
import sys
import logging
import json
import csv

import six
from tqdm import tqdm
import tensorflow as tf

from ..datasets.basedataset import Dataset
from ..models.basemodel import Model
from ..metrics.basemetrics import Metric
from ..utils import TF_DTYPES, get_checkpoints


class Evaluator(object):
    def __init__(self, config, path):
        self.config = config
        self.path = path

        self._configure_logger()

        ckpt_dir = config['other']['checkpoints_dir']
        self.checkpoints_dir = os.path.join(
            path, ckpt_dir)

        # Check if model checkpoint exists
        ckpt = get_checkpoints(self.checkpoints_dir)
        if ckpt is None:
            self.logger.warning(
                'No checkpoint was found',
                extra={'phase': 'construction'})
            return None
        else:
            ckpt_type, ckpt_path, ckpt_step = ckpt
            self._ckpt_type = ckpt_type
            self._ckpt_path = ckpt_path
            self._ckpt_step = ckpt_step

        evals_dir = config['evaluations']['evaluations_dir']
        self.evaluations_dir = os.path.join(path, evals_dir)

        if not os.path.exists(self.evaluations_dir):
            os.makedirs(self.evaluations_dir)

        fmt = self.config['evaluations']['results_format']
        filepath = os.path.join(
            self.evaluations_dir,
            'evaluation_step_{}.{}'.format(self._ckpt_step, fmt))

        if os.path.exists(filepath):
            msg = 'An evaluation file at step {} already exists.'
            msg = msg.format(self._ckpt_step)
            raise ValueError(msg)

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        log_config = self.config['logging']

        if not log_config.getboolean('logging'):
            logger.disable(logging.INFO)
            self.logger = logger
            return None

        log_format = '%(levelname)s: [%(asctime)-15s] [%(phase)s] %(message)s'
        formatter = logging.Formatter(log_format)

        verbosity = log_config.getint('verbosity')
        if verbosity == 1:
            level = logging.ERROR
        elif verbosity == 2:
            level = logging.WARNING
        elif verbosity == 3:
            level = logging.INFO
        elif verbosity == 4:
            level = logging.DEBUG
        else:
            msg = 'Verbosity level {l} is not in [1, 2, 3, 4]'
            raise ValueError(msg.format(verbosity))
        logger.setLevel(level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        if log_config.getboolean('log_to_file'):
            path = os.path.join(self.path, log_config['log_path'])
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logger

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

    def _save_evaluations(self, evaluations):
        path = os.path.join(
            self.evaluations_dir,
            'evaluation_step_{}'.format(self._ckpt_step))

        file_format = self.config['evaluations']['results_format']
        if file_format == 'json':
            with open(path + '.json', 'w') as jsonfile:
                json.dump(evaluations, jsonfile)

        elif file_format == 'csv':
            with open(path + '.csv', 'w') as csvfile:
                fieldnames = evaluations[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in evaluations:
                    writer.writerow(row)

    def _evaluate_tf(self, model=None, dataset=None, metrics=None):
        # Check for correct types
        assert issubclass(model, Model)
        assert issubclass(dataset, Dataset)
        assert isinstance(metrics, (list, tuple))

        for metric in metrics:
            assert issubclass(metric, Metric)

        # Instantiate metrics
        metrics = [metric() for metric in metrics]

        # Create new graph for model evaluation
        self.logger.info(
            'Building model and dataset',
            extra={'phase': 'construction'})
        graph = tf.Graph()

        # Instantiate model with graph
        model_instance = model(graph=graph)

        # Create input pipeline
        with graph.as_default():
            dataset_instance = dataset()
            ids, inputs, labels = dataset_instance.iter_test()

        # Add an extra dimension for batch
        prediction_tensor = model_instance.predict(tf.expand_dims(inputs, 0))

        self.logger.info(
                'Starting session and restoring model',
                extra={'phase': 'construction'})
        sess = tf.Session(graph=graph)
        if self._ckpt_type == 'tf':
            model_instance.restore(sess, self._ckpt_path)
        elif self._ckpt_type == 'numpy':
            model_instance.numpy_restore(sess, self._ckpt_path)

        self.logger.info(
                'Starting evaluation',
                extra={'phase': 'construction'})
        evaluations = []
        pbar = tqdm()
        try:
            while True:
                id_, prediction, label = sess.run(
                    [ids, prediction_tensor, labels])

                # Remove extra dimension from batch=1
                prediction = prediction[0]

                results = {'id': id_}
                for metric in metrics:
                    results.update(metric(prediction, label))

                evaluations.append(results)

                if self.config['evaluations'].getboolean('save_predictions'):
                    self._save_prediction(id_, prediction)

                pbar.update(1)

        except tf.errors.OutOfRangeError:
            self.logger.info(
                'Iterations over',
                extra={'phase': 'evaluations'})

        if self.config['evaluations'].getboolean('save_results'):
            self.logger.info(
                    'Saving results',
                    extra={'phase': 'saving'})
            self._save_evaluations(evaluations)

        return evaluations

    def _evaluate_no_tf(self, model=None, dataset=None, metrics=None):
        # Check for correct types
        assert issubclass(model, Model)
        assert issubclass(dataset, Dataset)
        assert isinstance(metrics, (list, tuple))

        for metric in metrics:
            assert issubclass(metric, Metric)

        # Instantiate metrics
        metrics = [metric() for metric in metrics]

        # Create new graph for model evaluation
        self.logger.info(
            'Building model and dataset',
            extra={'phase': 'construction'})
        graph = tf.Graph()

        # Instantiate model with graph
        model_instance = model(graph=graph)

        # Create input pipeline
        with graph.as_default():
            dataset_instance = dataset()
            input_tensors = self._build_inputs(dataset)

        prediction_tensor = model_instance.predict(input_tensors)

        self.logger.info(
            'Starting session and restoring model',
            extra={'phase': 'construction'})
        sess = tf.Session(graph=graph)
        if self._ckpt_type == 'tf':
            model_instance.restore(sess, self._ckpt_path)
        elif self._ckpt_type == 'numpy':
            model_instance.numpy_restore(sess, self._ckpt_path)

        self.logger.info(
            'Starting evaluation',
            extra={'phase': 'construction'})
        evaluations = []
        for id_, inputs, label in tqdm(dataset_instance.iter_test()):
            feed_dict = self._make_feed_dict(input_tensors, inputs)
            prediction = sess.run(prediction_tensor, feed_dict=feed_dict)

            results = {'id': id_}
            for metric in metrics:
                results.update(metric(prediction, label))

            evaluations.append(results)

            if self.config['evaluations'].getboolean('save_predictions'):
                self._save_prediction(id_, prediction)

        if self.config['evaluations'].getboolean('save_results'):
            self.logger.info(
                'Saving results',
                extra={'phase': 'saving'})
            self._save_evaluations(evaluations)

        return evaluations

    def evaluate(self, model=None, dataset=None, metrics=None):
        assert issubclass(dataset, Dataset)

        # Check if dataset is a tf dataset
        with tf.Graph().as_default():
            dataset_instance = dataset()
            is_tf = isinstance(dataset_instance.iter_test()[0], tf.Tensor)

        if is_tf:
            self.logger.info(
                'Tensorflow dataset detected.',
                extra={'phase': 'construction'})
            evaluations = self._evaluate_tf(
                model=model, dataset=dataset, metrics=metrics)
        else:
            self.logger.info(
                'Iterable dataset detected.',
                extra={'phase': 'construction'})
            evaluations = self._evaluate_no_tf(
                model=model, dataset=dataset, metrics=metrics)

        return evaluations
