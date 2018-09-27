from abc import ABCMeta, abstractmethod


class Model(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(inputs):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def restore(self, path):
        pass
