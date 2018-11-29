from abc import ABCMeta, abstractmethod
import six
import logging


@six.add_metaclass(ABCMeta)
class Product(object):
    logger = logging.getLogger(__name__)

    @abstractmethod
    def read_file(self, filepath):
        pass

    def preprocess(self, inputs):
        return inputs

    @abstractmethod
    def process(self, preprocessed_inputs):
        pass

    def _process_file(self, filepath):
        self.logger.info('Reading file %s', filepath)
        input_ = self.read_file(filepath)

        self.logger.info('Preprocessing file')
        preprocessed_input = self.preprocess(input_)

        self.logger.info('Processing file')
        output = self.process(preprocessed_input)

        self.logger.info('Done processing file %s', filepath)
        return output
