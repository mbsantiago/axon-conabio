from abc import ABCMeta, abstractmethod
import uuid

import six
import tensorflow as tf

from ..utils import summaries_scope


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
        with summaries_scope(self.summaries):
            with self.graph.as_default():
                with tf.name_scope(self.name):
                    loss = self._build_loss(outputs, labels)

                summaries = tf.get_collection(self.id)
        return loss, summaries

    def build_model_loss(self, model, inputs, labels, num_gpus=1):
        if num_gpus == 1:
            model_outputs = model.predict(inputs)
            loss, summaries = self.build_single_loss(model_outputs, labels)
            losses = [loss]

        else:
            losses = []

            split_inputs = tf.split(inputs, num_gpus)
            split_labels = tf.split(labels, num_gpus)
            for i in range(num_gpus):
                with tf.name_scope('tower_{}'.format(i)):
                    with tf.device('/device:GPU:{}'.format(i)):
                        model_outputs = model.predict(split_inputs[i])
                        loss, summs = self.build_single_loss(
                            model_outputs,
                            split_labels[i])

                        losses.append(loss)

        return losses

    def summary_op(self, prefix=None):
        """Return merged summary operation for all summaries defined within."""
        with self.graph.as_default():
            if prefix is None:
                scope_name = '{name}/'.format(name=self.name)
            else:
                scope_name = '{prefix}/{name}/'.format(
                    name=self.name, prefix=prefix)

            with tf.name_scope(scope_name):
                summaries = []

                for key in self.summaries:
                    name, func_name = key

                    tensors = []

                    # Since all summaries are of the same type and have the
                    # same name, it can be assumed that all summaries have been
                    # called with the same arguments, hence we can take wlog
                    # the first arguments.
                    summary_function = self.summaries[key][0][0]
                    global_arguments = self.summaries[key][0][1][2:]
                    global_kwargs = self.summaries[key][0][2]

                    for func, args, kwargs in self.summaries[key]:
                        # Tensor argument is always the second argument in
                        # tensorflow summary functions. Should this change this
                        # part of the code may break.
                        tensor = args[1]
                        tensors.append(tensor)

                    # Aggregation is different in the scalar case since it is
                    # assumed to be a tensor containing a single value.
                    scope_name = 'summary_aggregation/{type}/{name}'
                    scope_name = scope_name.format(type=func_name, name=name)
                    with tf.name_scope(scope_name):
                        if func_name == 'scalar':
                            ntensors = len(tensors)
                            aggregated_tensors = tf.add_n(tensors) / ntensors
                        else:
                            aggregated_tensors = tf.concat(tensors, axis=0)

                    # Add first two arguments to arguments list
                    global_arguments = (
                        [name, aggregated_tensors] +
                        list(global_arguments))

                    summary = summary_function(
                        *global_arguments,
                        **global_kwargs)
                    summaries.append(summary)

                summary_op = tf.summary.merge(summaries)
        return summary_op

    @abstractmethod
    def _build_loss(self, outputs, labels):
        pass
