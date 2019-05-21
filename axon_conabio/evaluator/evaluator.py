import os
import sys
import logging
import json
import csv

import six
from tqdm import tqdm
import tensorflow as tf

from axon_conabio.datasets.basedataset import Dataset
from axon_conabio.models.basemodel import Model
from axon_conabio.metrics.basemetrics import Metric
from axon_conabio.management.utils import get_model_checkpoint

from axon_conabio.utils import TF_DTYPES
from axon_conabio.axon_logging import BaseLogger
from axon_conabio.writers import FileWriter


class EvaluationError(Exception):
    pass


class Evaluator(object):
    logger_class = BaseLogger
    stopping_errors = (
        Exception,
    )

    @classmethod
    def get_evaluator(cls, dataset):
        is_tf = check_for_tf_dataset(dataset)

        if is_tf:
            return TFEvaluator
        else:
            return FeedDictEvaluator

    def __init__(self, config, path):
        self.config = config
        self.path = path

        self.configure_logger()
        self.configure_checkpoints()
        self.configure_evaluations_directory()

    def evaluate(
            self,
            model=None,
            dataset=None,
            metrics=None,
            name=None,
            dataset_kwargs=None,
            model_kwargs=None):
        self.check_evaluation_inputs(model, dataset, metrics)
        self.check_if_evaluation_file_exists(name)

        if model_kwargs is None:
            model_kwargs = {}

        if dataset_kwargs is None:
            dataset_kwargs = {}

        graph = tf.Graph()

        metrics = [metric() for metric in metrics]

        self.logger.info(
            'Building input pipeline',
            extra={'phase': 'construction'})
        iterator, tensors = self.create_input_pipeline(
            graph,
            model,
            dataset,
            dataset_kwargs)

        self.logger.info(
            'Building model graph',
            extra={'phase': 'construction'})
        model_instance = model(graph=graph, **model_kwargs)
        prediction_tensor = self.build_prediction_tensor(model_instance, tensors)

        self.logger.info(
            'Starting session and restoring model',
            extra={'phase': 'construction'})
        sess = self.start_session(graph, config, model_instance)

        self.logger.info(
                'Starting evaluation',
                extra={'phase': 'construction'})
        evaluations = []

        pbar = tqdm()
        try:
            while True:
                prediction, label = self.evaluation_step(
                    sess,
                    prediction_tensor,
                    iterator,
                    tensors)
                results = self.get_metric_results(prediction, label)

                evaluations.append(results)

                if bool(self.config['evaluations']['save_predictions']):
                    self.save_prediction(results, prediction)

                pbar.update(1)

        except self.stopping_errrors:
            self.logger.info(
                'Evaluation over',
                extra={'phase': 'evaluations'})

            if bool(self.config['evaluations']['save_results']):
                self.logger.info(
                    'Saving results',
                    extra={'phase': 'saving'})
                self.save_evaluations(evaluations, name=name)

        pbar.close()
        return evaluations

    @abstractmethod
    def build_prediction_tensor(self, model_instance, tensors):
        pass

    @abstractmethod
    def create_input_pipeline(self, graph, model, dataset, dataset_kwargs):
        pass

    @abstractmethod
    def evaluation_step(self, sess, prediction_tensor, iterator, tensors):
        pass

    def start_session(self, graph, config, model_instance):
        config = tf.ConfigProto(device_count={'GPU': 0})
        sess = tf.Session(graph=graph, config=config)

        with graph.as_default():
            tables_init = tf.tables_initializer()
            local_init = tf.local_variables_initializer()

        sess.run([tables_init, local_init])
        sess.run(model_instance.init_op())

        model_instance.restore(
            sess, path=self._ckpt_path, mode=self._ckpt_type)

        return sess

    def get_metric_results(self, prediction, label):
        # Remove extra dimension from batch=1
        prediction = prediction[0]

        metadata = self.get_object_metadata(label)
        results = metadata

        for metric in metrics:
            results.update(metric(prediction, label))

        return results

    def get_object_metadata(self, label):
        return {'id': label['id']}

    def configure_checkpoints(self):
        ckpt_dir = self.config['other']['checkpoints_dir']
        self.checkpoints_dir = os.path.join(path, ckpt_dir)

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

    def configure_evaluations_directory(self):
        evals_dir = self.config['evaluations']['evaluations_dir']
        self.evaluations_dir = os.path.join(path, evals_dir)

        if not os.path.exists(self.evaluations_dir):
            os.makedirs(self.evaluations_dir)

    def configure_logger(self):
        logger = self.logger_class(
            name=__name__,
            config=self.config['logging'],
            path=self.path)
        self.logger = logger.get_logger()

    def save_evaluations(self, evaluations, name=''):
        path = os.path.join(
            self.evaluations_dir,
            'evaluation_{}_step_{}'.format(name, self._ckpt_step))

        file_format = self.config['evaluations']['results_format']
        writer = FileWriter.get_writer(file_format)

        writer.save(path, evaluations)

    def check_if_evaluation_file_exists(self, name):
        fmt = self.config['evaluations']['results_format']
        filepath = os.path.join(
                self.evaluations_dir,
                'evaluation_{}_step_{}.{}'.format(name, self._ckpt_step, fmt))

        if os.path.exists(filepath):
            msg = 'An evaluation file at step {} already exists. Skipping.'
            msg = msg.format(self._ckpt_step)
            self.logger.warning(msg, extra={'phase': 'evaluation'})
            raise EvaluationError(msg)

    def check_evaluation_inputs(self, model, dataset, metrics):
        assert issubclass(dataset, Dataset)
        assert issubclass(model, Model)
        assert isinstance(metrics, (list, tuple))

        for metric in metrics:
            assert issubclass(metric, Metric)


class TFEvaluator(Evaluator):
    stopping_errors = (
        tf.errors.OutOfRangeError,
    )

    def create_prediction_tensor(self, model_instance, tensors)
        inputs, labels = tensors
        return model_instance.predict(inputs)

    def create_input_pipeline(self, graph, model, dataset, dataset_kwargs):
        with graph.as_default():
            dataset_instance = dataset(**dataset_kwargs)
            tensors = dataset_instance.iter_test()

        return None, tensors

    def evaluation_step(self, sess, prediction_tensor, iterator, tensors):
        _, labels = tensors
        return sess.run([prediction_tensor, labels])


class FeedDictEvaluator(Evaluator):
    stopping_errors = (
        StopIteration,
        IndexError,
    )

    def create_prediction_tensor(self, model_instance, tensors)
        return model_instance.predict(tensors)

    def create_input_pipeline(self, graph, model, dataset, dataset_kwargs):
        with graph.as_default():
            dataset_instance = dataset(**dataset_kwargs)
            iterator = dataset_instance.iter_test()
            tensors = self.build_inputs(model)

        return iterator, tensors

    def evaluation_step(self, sess, prediction_tensor, iterator, tensors):
        inputs, label = iterator.next()
        feed_dict = self.make_feed_dict(tensors, inputs)
        prediction = sess.run(prediction_tensor, feed_dict=feed_dict)
        return prediction, label

    def build_inputs(self, model):
        input_structure = model.input_structure

        inputs = {
            key: tf.placeholder(get_dtype(value), shape=get_shape(value))
            for key, value in six.iteritems(input_structure)
        }

        if len(inputs) == 1:
            _, inputs = inputs.popitem()

        return inputs

    def make_feed_dict(self, input_tensors, inputs):
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


def check_for_tf_dataset(dataset):
    with tf.Graph().as_default():
        dataset_instance = dataset()
        try:
            is_tf = isinstance(dataset_instance.iter_test()[0], tf.Tensor)
        except TypeError:
            is_tf = False


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


def bla():
    is_tf = check_for_tf_dataset(dataset)

    if is_tf:
        self.logger.info(
                'Tensorflow dataset detected.',
                extra={'phase': 'construction'})
        return self._evaluate_tf(
                model=model,
                dataset=dataset,
                metrics=metrics,
                name=name,
                dataset_kwargs=dataset_kwargs)

    self.logger.info(
            'Iterable dataset detected.',
            extra={'phase': 'construction'})
    return self._evaluate_no_tf(
            model=model,
            dataset=dataset,
            metrics=metrics,
            name=name,
            dataset_kwargs=dataset_kwargs)
