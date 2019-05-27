from .base import BaseLogger


class DummyLogger(BaseLogger):
    def prepare_for_training(self):
        pass

    def add_logging_operations(self, operations, run=None, category=None):
        pass

    def check_and_log(self, step):
        pass
