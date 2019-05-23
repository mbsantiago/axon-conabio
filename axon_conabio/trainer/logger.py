from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class TrainLogger(object):
    def __init__(self, config, path):
        self.config = config
        self.path = path

    @abstractmethod
    def prepare_for_training(self, context):
        pass

    @abstractmethod
    def update_training_configurations(self, train_resources, train_outputs):
        pass

    @abstractmethod
    def update_validation_configuration(self, validation_resources):
        pass

    @abstractmethod
    def check_and_log(self, context, step):
        pass


class DummyTrainLogger(TrainLogger):
    def prepare_for_training(self, context):
        pass

    def update_training_configurations(self, train_resources, train_outputs):
        pass

    def update_validation_configuration(self, validation_resources):
        pass

    def check_and_log(self, context, log):
        pass
