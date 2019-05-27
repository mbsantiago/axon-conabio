from abc import abstractmethod


class Loss(object):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def description(self):
        pass

    def __init__(self, name=''):
        self.instance_name = name
        self.log_operations = {}

    @abstractmethod
    def build_loss(self, outputs, labels):
        pass

    def register_log_operation(self, operation, name):
        self.log_operations[name] = operation

    def get_summary_operations(self):
        return self.log_operations
