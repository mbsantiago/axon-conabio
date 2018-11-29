from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class Model(object):

    @abstractmethod
    def predict(self, inputs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass
