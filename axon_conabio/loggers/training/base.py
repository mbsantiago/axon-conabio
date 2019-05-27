from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class BaseLogger(object):
    def __init__(self, config, path):
        self.config = config
        self.path = path

    @abstractmethod
    def add_logging_operations(self, operations, run=None, category=None):
        pass

    @abstractmethod
    def check_and_log(self, step):
        pass
