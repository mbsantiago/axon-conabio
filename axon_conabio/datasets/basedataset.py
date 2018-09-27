from abc import ABC


class Dataset(ABC):

    def iter_train(self):
        pass

    def iter_validation(self):
        pass

    def iter_test(self):
        pass
