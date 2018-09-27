from abc import ABCMeta, abstractmethod


class Model(object):
    __metaclass__ = ABCMeta

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def predict(inputs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass
