from abc import ABCMeta, abstractmethod
from functools import partial
import six

import tensorflow as tf


from ..utils import summary_scope, summary_aggregation


@six.add_metaclass(ABCMeta)
class TrainLogger(object):
    def __init__(self, graph=None):
        if graph is None:
            graph = tf.get_default_graph()
        self.graph = graph

        self.tensors = {}

    @abstractmethod
    def _log(self, model, loss, inputs, labels):
        pass

    def log(self, model, loss, inputs, labels, run_name=None):
        summaries = {}
        self.tensors = {}

        with summary_scope(summaries):
            with self.graph.as_default():
                tmp = model.get_tensor
                tmp2 = model.get_tensors

                model.get_tensor = partial(
                    model.get_tensor, run_name=run_name)
                model.get_tensors = partial(
                    model.get_tensors, run_name=run_name)

                self._log(model, loss, inputs, labels)

                model.get_tensor = tmp
                model.get_tensors = tmp2

        summaries = summary_aggregation(summaries)
        return summaries, self.tensors

    def log_tensor(self, tensor_name, tensor):
        if tensor_name in self._tmp_tensor_storage:
            msg = 'Tensor name previously logged: {name}'
            msg = msg.format(name=tensor_name)
            raise KeyError(msg)

        self._tmp_tensor_storage[tensor_name] = tensor

    def get_extra_summaries(self):
        pass
