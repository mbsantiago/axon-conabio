from abc import ABC, abstractmethod
import tensorflow as tf

from .basemodel import Model


class TFModel(Model, ABC):
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

    def add_variables(self, variables):
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

    @abstractmethod
    def _predict(self, inputs):
        pass

    def predict(self, inputs):
        with self.graph.as_default():
            if self.submodel:
                vscope_name = self.name
            else:
                vscope_name = 'variables/{name}'.format(name=self.name)

            with tf.variable_scope(
                    vscope_name,
                    reuse=tf.AUTO_REUSE,
                    auxiliary_name_scope=False) as scope:
                with tf.name_scope(self.name):
                    results = self._predict(inputs)

                    variables = (
                        scope.trainable_variables() +
                        scope.local_variables() +
                        scope.global_variables()
                    )

                    self.add_variables(variables)

        return results

    def save(self, sess, path, **kwargs):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(self.variables)

        self.saver.save(sess, path, **kwargs)

    def restore(self, sess, path):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver(self.variables)

        self.saver.restore(sess, path)

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
