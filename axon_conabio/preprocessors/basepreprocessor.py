from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class Preprocessor(object):

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def preprocess(self, inputs):
        pass

    def __call__(self, inputs):
        return self.preprocess(inputs)
