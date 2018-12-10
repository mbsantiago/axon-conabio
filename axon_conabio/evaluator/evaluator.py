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
from ..management.utils import get_model_checkpoint
from ..utils import TF_DTYPES


class Evaluator(object):
    def __init__(self, config, path, ckpt=None):
        self.config = config
        self.path = path

        self._configure_logger()

        ckpt_dir = config['other']['checkpoints_dir']
        self.checkpoints_dir = os.path.join(
            path, ckpt_dir)

        # Check if model checkpoint exists
        try:
            ckpt_type, ckpt_path, ckpt_step = get_model_checkpoint(
                os.path.basename(path), ckpt=ckpt)

            self._ckpt_type = ckpt_type
            self._ckpt_path = ckpt_path
            self._ckpt_step = ckpt_step

        except (IOError, RuntimeError):
            self.logger.warning(
                'No checkpoint was found',
                extra={'phase': 'construction'})

        evals_dir = config['evaluations']['evaluations_dir']
        self.evaluations_dir = os.path.join(path, evals_dir)

        if not os.path.exists(self.evaluations_dir):
            os.makedirs(self.evaluations_dir)

    def _configure_logger(self):
        logger = logging.getLogger(__name__)
        log_config = self.config['logging']

        if not bool(log_config['logging']):
            logger.disable(logging.INFO)
            self.logger = logger
            return None

        log_format = '%(levelname)s: [%(asctime)-15s] [%(phase)s] %(message)s'
        formatter = logging.Formatter(log_format)

        verbosity = int(log_config['verbosity'])
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

        if bool(log_config['log_to_file']):
            path = os.path.join(self.path, log_config['log_path'])
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def _build_inputs(self, model):
        input_structure = model.input_structure

        def get_dtype(args):
            if not isinstance(args, tuple):
                return tf.float32

            if len(args) == 1:
                return tf.float32
            else:
                dtype_string = args[1]
                return TF_DTYPES[dtype_string]

        def get_shape(args):
            if not isinstance(args, tuple):
                return args
            else:
                return args[0]

        inputs = {
            key: tf.placeholder(get_dtype(value), shape=get_shape(value))
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
                feed_dict[value] = input_value

        # Case for single input
        else:
            feed_dict[input_tensors] = inputs

        return feed_dict

    def _save_evaluations(self, evaluations, name=''):
        path = os.path.join(
            self.evaluations_dir,
            'evaluation_{}_step_{}'.format(name, self._ckpt_step))

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

    def _evaluate_tf(
            self,
            model=None,
            model_kwargs=None,
            dataset=None,
            dataset_kwargs=None,
            metrics=None,
            name=None):
        # Check for correct types
        assert issubclass(model, Model)
        assert issubclass(dataset, Dataset)
        assert isinstance(metrics, (list, tuple))

        for metric in metrics:
            assert issubclass(metric, Metric)

        if model_kwargs is None:
            model_kwargs = {}

        if dataset_kwargs is None:
            dataset_kwargs = {}

        # Instantiate metrics
        metrics = [metric() for metric in metrics]

        # Create new graph for model evaluation
        self.logger.info(
            'Building model and dataset',
            extra={'phase': 'construction'})
        graph = tf.Graph()

        # Instantiate model with graph
        model_instance = model(graph=graph, **model_kwargs)

        # Create input pipeline
        with graph.as_default():
            dataset_instance = dataset(**dataset_kwargs)
            ids, inputs, labels = dataset_instance.iter_test()

        prediction_tensor = model_instance.predict(inputs)

        self.logger.info(
            'Starting session and restoring model',
            extra={'phase': 'construction'})
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(graph=graph, config=config)
        with graph.as_default():
            tables_init = tf.tables_initializer()
            local_init = tf.local_variables_initializer()
        sess.run([tables_init, local_init])
        sess.run(model_instance.init_op())
        model_instance.restore(
            sess, path=self._ckpt_path, mode=self._ckpt_type)

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

                results = {'id': id_[0]}
                for metric in metrics:
                    results.update(metric(prediction, label))

                evaluations.append(results)

                if bool(self.config['evaluations']['save_predictions']):
                    self._save_prediction(id_, prediction)

                pbar.update(1)

        except tf.errors.OutOfRangeError:
            self.logger.info(
                'Iterations over',
                extra={'phase': 'evaluations'})

        if bool(self.config['evaluations']['save_results']):
            self.logger.info(
                    'Saving results',
                    extra={'phase': 'saving'})
            self._save_evaluations(evaluations, name=name)

        pbar.close()
        return evaluations

    def _evaluate_no_tf(
            self,
            model=None,
            dataset=None,
            metrics=None,
            dataset_kwargs=None,
            model_kwargs=None,
            name=None):
        # Check for correct types
        assert issubclass(model, Model)
        assert issubclass(dataset, Dataset)
        assert isinstance(metrics, (list, tuple))

        for metric in metrics:
            assert issubclass(metric, Metric)

        if dataset_kwargs is None:
            dataset_kwargs = {}

        if model_kwargs is None:
            model_kwargs = {}

        # Instantiate metrics
        metrics = [metric() for metric in metrics]

        # Create new graph for model evaluation
        self.logger.info(
            'Building model and dataset',
            extra={'phase': 'construction'})
        graph = tf.Graph()

        # Instantiate model with graph
        model_instance = model(graph=graph, **model_kwargs)

        # Create input pipeline
        with graph.as_default():
            dataset_instance = dataset(**dataset_kwargs)
            input_tensors = self._build_inputs(model)

        prediction_tensor = model_instance.predict(input_tensors)

        self.logger.info(
            'Starting session and restoring model',
            extra={'phase': 'construction'})
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(graph=graph, config=config)
        with graph.as_default():
            tables_init = tf.tables_initializer()
            local_init = tf.local_variables_initializer()
        sess.run([tables_init, local_init])
        sess.run(model_instance.init_op())
        model_instance.restore(
            sess, path=self._ckpt_path, mode=self._ckpt_type)

        self.logger.info(
            'Starting evaluation',
            extra={'phase': 'construction'})
        evaluations = []
        for id_, inputs, label in tqdm(dataset_instance.iter_test()):
            try:
                feed_dict = self._make_feed_dict(input_tensors, inputs)
                prediction = sess.run(prediction_tensor, feed_dict=feed_dict)

                results = {'id': id_}
                for metric in metrics:
                    results.update(metric(prediction, label))

                evaluations.append(results)

                if bool(self.config['evaluations']['save_predictions']):
                    self._save_prediction(id_, prediction)

            except Exception as exc:
                self.logger.warning(
                    str(exc),
                    extra={'phase': 'evaluation'})

        if bool(self.config['evaluations']['save_results']):
            self.logger.info(
                'Saving results',
                extra={'phase': 'saving'})
            self._save_evaluations(evaluations, name=name)

        return evaluations

    def evaluate(
            self,
            model=None,
            dataset=None,
            metrics=None,
            name=None,
            dataset_kwargs=None):
        fmt = self.config['evaluations']['results_format']
        filepath = os.path.join(
            self.evaluations_dir,
            'evaluation_{}_step_{}.{}'.format(name, self._ckpt_step, fmt))

        if os.path.exists(filepath):
            msg = 'An evaluation file at step {} already exists. Skipping.'
            msg = msg.format(self._ckpt_step)
            self.logger.warning(msg, extra={'phase': 'evaluation'})
            return []

        assert issubclass(dataset, Dataset)

        # Check if dataset is a tf dataset
        with tf.Graph().as_default():
            dataset_instance = dataset()
            try:
                is_tf = isinstance(dataset_instance.iter_test()[0], tf.Tensor)
            except TypeError:
                is_tf = False

        if is_tf:
            self.logger.info(
                'Tensorflow dataset detected.',
                extra={'phase': 'construction'})
            evaluations = self._evaluate_tf(
                model=model,
                dataset=dataset,
                metrics=metrics,
                name=name,
                dataset_kwargs=dataset_kwargs)
        else:
            self.logger.info(
                'Iterable dataset detected.',
                extra={'phase': 'construction'})
            evaluations = self._evaluate_no_tf(
                model=model,
                dataset=dataset,
                metrics=metrics,
                name=name,
                dataset_kwargs=dataset_kwargs)

        return evaluations
