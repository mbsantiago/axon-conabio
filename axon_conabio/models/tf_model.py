from abc import ABCMeta, abstractmethod
import tensorflow as tf
from uuid import uuid4

from .basemodel import Model


class TFModel(Model):
    __metaclass__ = ABCMeta

    def __init__(self, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph

        self.id = str(uuid4())

        # Make global step variable
        with self.graph.as_default():
            with tf.variable_scope(
                    'variables/{id}'.format(id=self.id),
                    reuse=tf.AUTO_REUSE,
                    auxiliary_name_scope=False):
                self.global_step = tf.get_variable(
                    'global_step',
                    shape=[1],
                    initializer=tf.zeros_initializer,
                    trainable=False)

                # Add to saveable variables
                self.variables = {'global_step': self.global_step}

        self.variables_are_set = False

    def add_variables(self, variables):
        if not self.variables_are_set:
            def parse_name(variable):
                name = variable.name
                name = '/'.join(name.split('/')[2:])
                name = name.split(':')[0]
                return name

            # Remove id from variable name
            variable_dict = {
                parse_name(variable): variable
                for variable in variables
            }

            print(variable_dict)

            # Add to saveable variables
            self.variables.update(variable_dict)
            self.variables_are_set = True

    @abstractmethod
    def _predict(self, inputs):
        pass

    def predict(self, inputs):
        with self.graph.as_default():
            with tf.variable_scope(
                    'variables/{id}'.format(id=self.id),
                    reuse=tf.AUTO_REUSE,
                    auxiliary_name_scope=False) as scope:
                results = self._predict(inputs)

                variables = (
                    scope.trainable_variables() +
                    scope.local_variables() +
                    scope.global_variables()
                )

                print('variables', variables)
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

    def init_op(self):
        with self.graph.as_default():
            return tf.global_variables_initializer()
