from abc import ABCMeta, abstractmethod


class Data(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def features(self):
        pass


class Label(Data):
    pass
