from abc import abstractmethod
import collections
import os
import shutil

from axon_conabio.models.tf_model import Model
from axon_conabio.losses.baseloss import Loss
from axon_conabio.datasets.basedataset import Dataset
from axon_conabio.axon_logging import BaseLogger
from axon_conabio.trainer.tf_trainer_config import get_config, TrainConfig
from axon_conabio.trainer.logger import DummyTrainLogger


TrainResources = collections.namedtuple('Pipeline', [
    'dataset',
    'inputs',
    'iterator',
    'loss',
    'regularization_loss',
])


class Trainer(object):
    logger_class = BaseLogger
    train_logger_class = DummyTrainLogger
    stop_errors = (StopIteration,)

    base_model_class = Model
    base_loss_class = Loss
    base_dataset_class = Dataset

    def __init__(self, config, model_config, path, retrain=False):
        if not isinstance(config, TrainConfig):
            config = get_config(config=config)

        self.config = config.config
        self.model_config = model_config
        self.optimizer_config = config.optimizer
        self.path = path
        self.retrain = retrain

        self.context = self.get_train_context()
        self.logger = self.get_logger()
        self.train_logger = self.get_training_logger()

        self.checkpoints = self.config['checkpoints']['checkpoints']
        self.checkpoint_frequency = self.config['checkpoints']['frequency']

        self.log_train = self.config['logging']['train']
        self.log_train_frequency = self.config['logging']['train_frequency']

        self.log_validation = self.config['logging']['validation']
        self.log_validation_frequency = self.config['logging']['validation_frequency']

        self.stop_at_step = self.config['feed']['stop_at_step']

        if self.retrain:
            self.clean_directory_structure()
        self.build_directory_structure()

    def train(
            self,
            model,
            loss,
            train_dataset,
            valid_dataset=None):

        self.check_train_inputs(model, loss, train_dataset, valid_dataset)
        if model_kwargs is None:
            model_kwargs = {}

        if loss_kwargs is None:
            loss_kwargs = {}

        if dataset_kwargs is None:
            dataset_kwargs = {}

        model_instance = self.get_model_instance(model)

        train_resources = self.build_input_resources(
            train_dataset, loss, model_instance)

        train_operations, train_outputs = self.build_train_operations(
            model_instance, train_resources)

        self.train_logger.update_training_configurations(
            train_resources,
            train_outputs)

        validation_resources = None
        if self.config.validate:
            validation_resources = self.build_input_resources(
                valid_dataset, loss, model_instance)

            self.train_logger.update_validation_configuration(
                validation_resources)

        init_operations = self.build_initialization_operations(
            model_instance)

        self.intialize_training_session(init_operations)
        self.train_logger.prepare_for_training(self.context)

        self.restore_model(model_instance)

        self.main_loop(
            train_resources,
            train_operations,
            train_outputs,
            validation_resources=validation_resources)

    def main_loop(
            self,
            train_resources,
            train_operations,
            model_instance,
            validation_resources=None):
        step = None
        try:
            while True:
                loss, step = self.run_training_step(
                    train_operations,
                    train_resources,
                    model_instance)

                self.train_logger.check_and_log(self.context, step)

                self.check_and_log_train_step(loss, step)
                self.check_and_log_validation_step(validation_resources, step)
                self.check_and_save_checkpoints(model_instance, step)

                if step == self.stop_at_step:
                    raise StopIteration

        except KeyboardInterrupt:
            self.handle_keyboard_interrupt(model_instance, step)

        except self.stop_errors:
            self.handle_stop(model_instance, step)


    def build_input_resources(self, dataset_class, loss_class, model_instance):
        dataset_instance = self.get_dataset_instance(dataset_class)
        loss_instance = self.get_loss_instance(loss_class)

        iterator, inputs = self.build_iterator_and_inputs(dataset_instance)
        outputs = self.build_outputs(model_instance, inputs)

        loss = self.build_loss(loss_instance, inputs, outputs)
        regularization_loss = self.build_regularization_loss(outputs)

        resources = TrainResources(
            dataset=dataset_instance,
            iterator=iterator,
            inputs=inputs,
            loss=loss,
            regularization_loss=regularization_loss)

        return resources

    @abstractmethod
    def get_train_context(self):
        pass

    @abstractmethod
    def get_model_instance(self, model):
        pass

    @abstractmethod
    def get_dataset_instance(self, dataset):
        pass

    @abstractmethod
    def build_iterator_and_inputs(self, dataset_instance):
        pass

    @abstractmethod
    def build_loss(self, loss_instance, inputs, outputs):
        pass

    @abstractmethod
    def get_loss_instance(self, loss):
        pass

    @abstractmethod
    def build_outputs(self, model_instance, pipeline):
        pass

    @abstractmethod
    def build_regularization_loss(self, outputs):
        pass

    @abstractmethod
    def build_train_operations(self, model_instance, resources):
        pass

    @abstractmethod
    def build_initialization_operations(self, model_instance):
        pass

    @abstractmethod
    def intialize_training_session(self, init_operations):
        pass

    @abstractmethod
    def restore_model(self, model_instance):
        pass

    @abstractmethod
    def save_checkpoint(self, model_instance, step):
        pass

    @abstractmethod
    def run_validation_step(self, validation_resources):
        pass

    @abstractmethod
    def run_training_step(self, train_operations, train_resources, model_instance):
        pass

    def handle_keyboard_interrupt(self, model_instance, step):
        msg = 'User interrupted training'
        self.logger.warning(msg, extra={'phase': 'end'})

        if step is not None:
            self.save_checkpoint(model_instance, step)

    def handle_stop(self, model_instance, step):
        msg = 'Dataset iterations done.'
        self.logger.info(msg, extra={'phase': 'end'})

        if step is not None:
            self.save_checkpoint(model_instance, step)

    def check_and_log_train_step(self, loss, step):
        if self.log_train and (step % self.log_train_frequency == 0):
            msg = '[step {st}] training loss : {ls}'.format(
                st=step, ls=loss)
            self.logger.info(msg, extra={'phase': 'training'})

    def check_and_log_validation_step(self, validation_resources, step):
        if self.log_validation and (step % self.log_validation_frequency == 0):
            validation_loss = self.run_validation_step(validation_resources)
            msg = '[step {st}] validation loss : {ls}'.format(
                st=step, ls=validation_loss)
            self.logger.info(msg, extra={'phase': 'training'})

    def check_and_save_checkpoints(self, model_instance, step):
        if self.checkpoints and (step % self.checkpoint_frequency == 0):
            self.save_checkpoint(model_instance, step)

    def check_path_exists(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def get_train_logging_directory(self):
        return os.path.join(
            self.path,
            self.config['summaries']['summaries_dir'])

    def get_checkpoints_directory(self):
        return os.path.join(
            self.path,
            self.config['checkpoints']['checkpoints_dir'])

    def clean_directory_structure(self):
        logging_dir = self.get_train_logging_directory()
        if os.path.exists(logging_dir):
            shutil.rmtree(logging_dir)

        checkpoints_dir = self.get_checkpoints_directory()
        if os.path.exists(checkpoints_dir):
            shutil.rmtree(checkpoints_dir)

    def build_directory_structure(self):
        logging_dir = self.get_train_logging_directory()
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        self.logging_dir = logging_dir

        checkpoints_dir = self.get_checkpoints_directory()
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        self.checkpoints_dir = checkpoints_dir

    def get_logger(self):
        logger = self.logger_class(
            name=__name__,
            config=self.config['logging'],
            path=self.path)
        return logger.get_logger()

    def get_training_logger(self):
        train_logger = self.train_logger_class(
            config=self.config['logging'],
            path=self.path)
        return train_logger

    def get_regularization_loss(self, model):
        regularizer = load_regularizer(self.config['regularization'])
        return regularizer.get_loss(model)

    def get_losses_instances(self, loss, loss_kwargs):
        train_loss = loss(context=self.context, **loss_kwargs)
        validation_loss = loss(context=self.context, **loss_kwargs)
        return train_loss, validation_loss

    @classmethod
    def check_train_inputs(cls, model, loss, train_dataset, valid_dataset):
        assert issubclass(model, cls.base_model_class)
        assert issubclass(loss, cls.base_loss_class)
        assert issubclass(train_dataset, cls.base_dataset_class)

        if valid_dataset is not None:
            assert issubclass(valid_dataset, cls.base_dataset_class)
