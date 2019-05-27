from abc import abstractmethod
import collections
import os
import re
import shutil

from axon_conabio.models.tf_model import Model
from axon_conabio.losses.baseloss import Loss
from axon_conabio.datasets.basedataset import Dataset
from axon_conabio.trainer.tf_trainer_config import get_config, TrainConfig

import axon_conabio.loggers as logging


TrainResources = collections.namedtuple('TrainResources', [
    'dataset',
    'inputs',
    'iterator',
    'loss',
    'regularization_loss',
])


class Trainer(object):
    logger_class = logging.console.BaseLogger
    train_logger_class = logging.training.DummyLogger
    stop_errors = (StopIteration,)

    base_model_class = Model
    base_loss_class = Loss
    base_dataset_class = Dataset

    def __init__(self, config, model_config, path, retrain=False):
        if not isinstance(config, TrainConfig):
            config = get_config(config=config)

        self.config = config.config
        self.model_config = model_config
        self.path = path
        self.retrain = retrain

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
            train_dataset, loss, model_instance, run='train')

        train_operations = self.build_train_operations(
            model_instance, train_resources)

        validation_resources = None
        if self.config.validate:
            validation_resources = self.build_input_resources(
                valid_dataset, loss, model_instance, run='validation')

        init_operations = self.build_initialization_operations(model_instance)

        self.intialize_training_session(init_operations)

        self.restore_model(model_instance)

        self.prepare_for_training_loop()
        self.main_loop(
            train_resources,
            train_operations,
            model_instance,
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

                self.check_and_log_train_logger(step)
                self.check_and_log_train_step(loss, step)
                self.check_and_log_validation_step(validation_resources, step)
                self.check_and_save_checkpoints(model_instance, step)

                if step == self.stop_at_step:
                    raise StopIteration

        except KeyboardInterrupt:
            self.handle_keyboard_interrupt(model_instance, step)

        except self.stop_errors:
            self.handle_stop(model_instance, step)


    def build_input_resources(self, dataset_class, loss_class, model_instance, run=None):
        dataset_instance = self.get_dataset_instance(dataset_class)
        iterator, inputs = self.build_iterator_and_inputs(dataset_instance)
        self.add_dataset_logging_operators(dataset_instance, run=run)

        outputs = self.build_outputs(model_instance, inputs)
        self.add_model_logging_operators(model_instance, run=run)

        loss_builder = self.get_loss_instance(loss_class)
        loss = self.build_loss(loss_builder, inputs, outputs)
        self.add_loss_logging_operators(loss_builder, run=run)

        regularizer = self.get_regularizer()
        regularization_loss = self.build_regularization_loss(regularizer, outputs)
        self.add_regularization_logging_operators(regularizer, run=run)

        resources = TrainResources(
            dataset=dataset_instance,
            iterator=iterator,
            inputs=inputs,
            loss=loss,
            regularization_loss=regularization_loss)

        return resources

    def get_regularizer(self):
        return 'TODO'

    def get_model_instance(self, model):
        model_kwargs = self.config.architecture.kwargs
        return model(**model_kwargs)

    def get_dataset_instance(self, dataset):
        dataset_kwargs = self.config.dataset.kwargs
        return dataset(**dataset_kwargs)

    @abstractmethod
    def build_iterator_and_inputs(self, dataset_instance):
        pass

    def build_loss(self, loss_builder, inputs, outputs):
        label = inputs['label']
        return loss_builder.build_loss(outputs, label)

    def get_loss_instance(self, loss):
        loss_kwargs = self.config.loss.kwargs
        return loss(**loss_kwargs)

    def build_outputs(self, model_instance, pipeline):
        outputs = model_instance.predict(pipeline.inputs)
        return outputs

    @abstractmethod
    def build_regularization_loss(self, regularizer, outputs):
        return regularizer.build_loss(outputs)

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
        path, step = self.get_latest_checkpoint_path()
        if path is not None:
            msg = 'Restoring model to step {}'.format(step)
            self.logger.info(msg, extra={'phase': 'initialization'})
            model_instance.restore(path)
        else:
            msg = ''

    @abstractmethod
    def save_checkpoint(self, model_instance, step):
        path = self.get_checkpoint_path(step)
        model_instance.save(path)

    @abstractmethod
    def run_validation_step(self, validation_resources):
        pass

    @abstractmethod
    def run_training_step(self, train_operations, train_resources, model_instance):
        pass

    def prepare_for_training_loop(self):
        pass

    def add_dataset_logging_operators(self, dataset_instance, run=None):
        dataset_logging_operations = dataset_instance.get_logging_operations()
        self.train_logger.add_logging_operations(
            dataset_logging_operations,
            run=run,
            category='dataset')

    def add_model_logging_operators(self, model_instance, run=None):
        model_logging_operations = model_instance.get_logging_operations()
        self.train_logger.add_logging_operations(
            model_logging_operations,
            run=run,
            category='architecture')

    def add_loss_logging_operators(self, loss_builder, run=None):
        loss_logging_operations = loss_builder.get_logging_operations()
        self.train_logger.add_logging_operations(
            loss_logging_operations,
            run=run,
            category='loss')

    def add_regularization_logging_operators(self, regularizer, run=None):
        regularization_logging_operations = regularizer.get_logging_operations()
        self.train_logger.add_logging_operations(
            regularization_logging_operations,
            run=run,
            category='regularization')

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

    def check_and_log_train_logger(self, step):
        self.train_logger.check_and_log(step)

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

    def get_checkpoint_path(self, step):
        checkpoints_dir = self.get_checkpoints_directory()
        return os.path.join(checkpoints_dir, 'step_{}.ckpt'.format(step))

    def get_latest_checkpoint_path(self):
        checkpoints_dir = self.get_checkpoints_directory()
        checkpoint, step = get_latest_checkpoint(checkpoints_dir)
        return checkpoint, step

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

    @classmethod
    def check_train_inputs(cls, model, loss, train_dataset, valid_dataset):
        assert issubclass(model, cls.base_model_class)
        assert issubclass(loss, cls.base_loss_class)
        assert issubclass(train_dataset, cls.base_dataset_class)

        if valid_dataset is not None:
            assert issubclass(valid_dataset, cls.base_dataset_class)



def get_latest_checkpoint(directory):
    files = os.listdir(directory)

    checkpoints = []
    for filename in files:
        match = re.match(r'step_([0-9]+).ckpt', filename)
        if match is not None:
            step = int(match.group(0))
            checkpoints.append((filename, step))

    if not checkpoints:
        return None, 0

    checkpoints.sort(key=lambda x: x[1])
    filename, step = checkpoints[-1]

    return os.path.join(directory, filename), step
