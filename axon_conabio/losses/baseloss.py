from abc import ABCMeta, abstractmethod
import uuid

import six
import tensorflow as tf

from ..utils import summary_scope, summary_aggregation


@six.add_metaclass(ABCMeta)
class Loss(object):
    def __init__(self, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph

        self.id = str(uuid.uuid4())
        self.summaries = {}

    @property
    @abstractmethod
    def name(self):
        pass

    def build_single_loss(self, outputs, labels):
        with summary_scope(self.summaries):
            with self.graph.as_default():
                with tf.name_scope(self.name):
                    loss = self._build_loss(outputs, labels)
        return loss

    def build_model_loss(
            self,
            model,
            inputs,
            labels,
            num_gpus=1,
            run_name=None):

        predict = model.predict

        if num_gpus == 1:
            model_outputs = predict(inputs, run_name=run_name)
            loss = self.build_single_loss(model_outputs, labels)
            losses = [loss]

        else:
            split_inputs = tf.split(inputs, num_gpus)
            split_labels = tf.split(labels, num_gpus)

            losses = []
            for i in range(num_gpus):
                with tf.device('/device:GPU:{}'.format(i)):
                    model_outputs = predict(
                        split_inputs[i],
                        run_name=run_name,
                        sub_run='sub_run_{}'.format(i))
                    loss = self.build_single_loss(
                        model_outputs,
                        split_labels[i])

                    losses.append(loss)

        return losses

    def summary_op(self, prefix=None):
        """Return merged summary operation for all summaries defined within."""
        if prefix is None:
            prefix = ''
        else:
            prefix += '/'

        with self.graph.as_default():
            summary_op = summary_aggregation(self.summaries, prefix=prefix)
        return summary_op

    @abstractmethod
    def _build_loss(self, outputs, labels):
        pass
