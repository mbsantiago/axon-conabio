from abc import ABCMeta, abstractmethod
import collections
import os

import tensorflow as tf
import numpy as np
import six

from .basemodel import Model
from ..utils import summary_scope, summary_aggregation


@six.add_metaclass(ABCMeta)
class TFModel(Model):
    @property
    @abstractmethod
    def name(self):
        return 'model'

    def __init__(self, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph

        # Count the number of times this model has appeared within the same
        # graph.
        if not hasattr(self.graph, 'model_counts'):
            self.graph.model_counts = {self.name: 1}
            self.submodel = False
        else:
            value = self.graph.model_counts.setdefault(self.name, 0)
            self.graph.model_counts[self.name] = value + 1
            self.submodel = True

        # Avoid name collision
        count = self.graph.model_counts[self.name]
        if count > 1:
            self.name = '{name}_{count}'.format(name=self.name, count=count)

        # Store reference to models in graph
        if not hasattr(self.graph, 'models'):
            self.graph.models = {self.name: self}
        else:
            self.graph.models[self.name] = self

        # Dictionary holding all created variables
        self.variables = {}
        self.variables_are_set = False

        # Storage for summaries operations within model
        self.summaries = {}

        # Storage for user-selected tensors
        self.tensors = {}
        self._tmp_tensor_storage = {}

        # Make global step variable
        if not self.submodel:
            with self.graph.as_default():
                with tf.variable_scope(
                        'variables/{name}'.format(name=self.name),
                        reuse=tf.AUTO_REUSE,
                        auxiliary_name_scope=False):
                    self.global_step = tf.get_variable(
                        'global_step',
                        dtype=tf.int64,
                        shape=[],
                        initializer=tf.zeros_initializer,
                        trainable=False)

                    # Add to saveable variables
                    self.variables['global_step'] = self.global_step

    def _add_variables(self, variables):
        if not self.variables_are_set:
            def parse_name(variable):
                name = variable.name
                name = '/'.join(name.split('/')[2:])
                name = name.split(':')[0]
                return name

            # Remove model name from variable name
            variable_dict = {
                parse_name(variable): variable
                for variable in variables
            }

            # Add to saveable variables
            self.variables.update(variable_dict)
            self.variables_are_set = True

    def register_tensor(self, tensor_name, tensor):
        if tensor_name in self._tmp_tensor_storage:
            msg = 'A tensor with name {name} has been previously registered'
            msg = msg.format(name=tensor_name)
            raise KeyError(msg)
        self._tmp_tensor_storage[tensor_name] = tensor

    def get_tensor(self, tensor_name, run_name=None):
        if run_name is None:
            run_name = 'default'

        tensors = self.tensors[run_name]

        if tensor_name not in tensors:
            msg = 'Tensor name {name} is not registered in model'
            raise KeyError(msg.format(name=tensor_name))

        tensor_list = tensors[tensor_name]

        if len(tensor_list) == 1:
            return tensor_list[0]
        else:
            return tf.concat(tensor_list, 0)

    def get_tensors(self, run_name=None):
        if run_name is None:
            run_name = 'default'

        tensors = {
            key: self.get_tensor(key, run_name=run_name)
            for key in self.tensors[run_name]}
        return tensors

    @abstractmethod
    def _predict(self, inputs):
        pass

    def predict(self, inputs, run_name=None, sub_run=None):
        # Empty temporal tensor storage
        self._tmp_tensor_storage = {}

        summaries = {}
        with summary_scope(summaries):
            with self.graph.as_default():
                if self.submodel:
                    vscope_name = self.name
                else:
                    vscope_name = 'variables/{name}'.format(name=self.name)

                with tf.variable_scope(
                        vscope_name,
                        reuse=tf.AUTO_REUSE,
                        auxiliary_name_scope=False) as scope:

                    scope_name = self.name
                    if run_name is not None:
                        scope_name += '/{}'.format(run_name)
                        if sub_run is not None:
                            scope_name += '/{}'.format(sub_run)

                    with tf.name_scope(scope_name):
                        results = self._predict(inputs)

                        variables = (
                            scope.trainable_variables() +
                            scope.local_variables() +
                            scope.global_variables()
                        )

                        self._add_variables(variables)

        self._register_tensors(run_name=run_name, sub_run=sub_run)
        self._register_summaries(summaries, run_name=run_name)
        return results

    def numpy_save(self, sess, path, step=None):
        variable_values = sess.run(self.variables)
        if step is not None:
            name = 'checkpoint_step_{}'.format(step)
        else:
            name = 'checkpoint'
        path = os.path.join(path, name)
        np.savez(path, **variable_values)

    def save(self, sess, path, **kwargs):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(self.variables)

        self.saver.save(sess, path, **kwargs)

    def restore(self, sess, path):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(self.variables)

        self.saver.restore(sess, path)

    def numpy_restore(self, sess, path):
        numpy_variables = np.load(path)

        for key in self.variables:
            var = self.variables[key]
            value = numpy_variables[key]

            var.load(value, sess)

    def _register_summaries(self, summaries, run_name=None):
        if run_name is None:
            run_name = 'default'

        if run_name not in self.summaries:
            self.summaries[run_name] = collections.defaultdict(list)

        for key in summaries:
            self.summaries[run_name][key].extend(summaries[key])

    def _register_tensors(self, run_name=None, sub_run=None):
        if run_name is None:
            run_name = 'default'

        if run_name not in self.tensors:
            self.tensors[run_name] = collections.defaultdict(list)

        for key in self._tmp_tensor_storage:
            self.tensors[run_name][key].append(
                self._tmp_tensor_storage[key])

    def get_model_summaries(self, run_name=None):
        if run_name is None:
            run_name = 'default'

        try:
            summaries = self.summaries[run_name]
        except KeyError:
            msg = 'No run with name {name} was registered.'
            raise KeyError(msg.format(name=run_name))

        prefix = '{name}/'.format(name=self.name)
        if run_name is not None:
            prefix += '{}/'.format(run_name)
        summary_op = summary_aggregation(summaries, prefix)
        return summary_op

    def get_variable_summaries(self, prefix=None):
        summaries = []
        if prefix is None:
            prefix = ''
        else:
            prefix += '/'

        for var_name in self.variables:
            if var_name == 'global_step':
                continue

            variable = self.variables[var_name]
            summaries.append(
                tf.summary.histogram(prefix + var_name, variable))
        return tf.summary.merge(summaries)

    def init_op(self):
        with self.graph.as_default():
            return tf.global_variables_initializer()
