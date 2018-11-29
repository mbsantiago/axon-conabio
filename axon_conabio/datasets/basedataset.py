from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class Dataset(object):
    @abstractmethod
    def iter_train(self):
        pass

    @abstractmethod
    def iter_validation(self):
        pass

    @abstractmethod
    def iter_test(self):
        pass
