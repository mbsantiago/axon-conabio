from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class Dataset(object):
    @property
    @abstractmethod
    def input_structure(self):
        pass

    def iter_train(self):
        pass

    def iter_validation(self):
        pass

    def iter_test(self):
        pass
