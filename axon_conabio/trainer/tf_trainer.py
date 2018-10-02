import logging
import collections
import os
import six
import sys
import tensorflow as tf
import numpy as np

from .tf_trainer_config import TrainerConfig
from ..models.tf_model import TFModel
from ..loss.baseloss import Loss
from ..datasets.basedataset import Dataset
from ..utils import get_checkpoints


logger = logging.getLogger(__name__)


class TFTrainer(object):
    def __init__(self, config, path):
        if not isinstance(config, TrainerConfig):
            config = TrainerConfig(config)
        self.config = config
        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)

        self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(__name__)

        if not self.config.logging:
            logger.disable(logging.INFO)
            self.logger = logger
            return None

        log_format = '%(levelname)s: [%(asctime)-15s] [%(phase)s] %(message)s'
        formatter = logging.Formatter(log_format)

        verbosity = self.config.verbosity
        if verbosity == 1:
            level = logging.ERROR
        elif verbosity == 2:
            level = logging.WARNING
        elif verbosity == 3:
            level = logging.INFO
        elif verbosity == 4:
            level = logging.DEBUG
        else:
            msg = 'Verbosity level {l} is not in [1, 2, 3, 4]'
            raise ValueError(msg.format(verbosity))
        logger.setLevel(level)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

        if self.config.log_to_file:
            path = os.path.join(self.path, self.config.log_path)
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def async_train(self, model=None, loss=None, dataset=None):
        # TODO
        pass

    def _get_regularization_loss(self, model):
        with model.graph.as_default():
            loss = 0
            if (self.config.l1_loss > 0 or self.config.l2_loss > 0):
                variables = [
                    variable for variable in model.variables.values()
                    if self._filter_regularization_variables(variable)]
                for var in variables:
                    if self.config.l1_loss > 0:
                        loss += tf.reduce_sum(tf.abs(var))

                    if self.config.l2_loss > 0:
                        loss += tf.nn.l2_loss(var)
        return loss

    def _filter_regularization_variables(self, variable):
        if 'bias' in variable.name:
            return False
        if not variable._trainable:
            return False
        return True

    def _get_optimizer(self):
        arguments = self.config.optimizer_arguments
        factory = self.config.optimizer_factory
        optimizer = factory(**arguments)
        return optimizer

    def _get_train_op_multiple_gpu(self, model, losses, reg_loss):
        num_gpus = self.config.num_gpus
        assert len(losses) == num_gpus

        with model.graph.as_default():
            optimizer = self._get_optimizer()
            gradients_list = []

            for i in range(num_gpus):
                with tf.device('/device:GPU:{}'.format(i)):
                    gradients = optimizer.compute_gradients(
                        losses[i] / num_gpus)
                    gradients_list += gradients

            if reg_loss != 0:
                gradients_list += optimizer.compute_gradients(reg_loss)

            train_op = optimizer.apply_gradients(
                gradients_list,
                global_step=model.global_step)

            total_loss = tf.add_n(losses) / num_gpus + reg_loss

        return train_op, total_loss, gradients_list

    def _build_summary_op(self, model, loss, gradients=None, prefix=None):
        summaries = [loss.summary_op(prefix=prefix)]

        if self.config.model_summaries:
            summaries.append(model.get_model_summaries(run_name=prefix))

        if self.config.variable_summaries:
            summaries.append(model.get_variable_summaries(prefix=prefix))

        if (self.config.gradient_summaries and gradients is not None):
            summaries.append(self._aggregate_gradients_summaries(gradients))

        return tf.summary.merge(summaries)

    def _aggregate_gradients_summaries(self, gradients):
        aggregated_gradients = collections.defaultdict(list)

        for gradient, var in gradients:
            aggregated_gradients[var.name].append(gradient)

        summaries = []
        for var, grad_list in six.iteritems(aggregated_gradients):
            name = var.split(':')[0]
            grad_list = [grad for grad in grad_list if grad is not None]
            if len(grad_list) == 0:
                continue
            elif len(grad_list) == 1:
                summaries.append(tf.summary.histogram(name, grad_list[0]))
            else:
                mean = tf.add_n(grad_list) / len(grad_list)
                summaries.append(tf.summary.histogram(name, mean))

        return tf.summary.merge(summaries)

    def _save_tensors(self, tensors, step):
        directory = os.path.join(
            self.path,
            self.config.tensors_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = os.path.join(
            directory,
            'step_{}'.format(step))

        num_sample = self.config.tensors_per_batch
        tensor_samples = {
            key: value[:num_sample]
            for key, value in six.iteritems(tensors)}
        np.savez(path, **tensor_samples)

    def train(self, model=None, loss=None, dataset=None):
        # Check if objects are elements of the corresponding classes
        assert issubclass(model, TFModel)
        assert issubclass(loss, Loss)
        assert issubclass(dataset, Dataset)

        # Configure logging
        tf.logging.set_verbosity(tf.logging.WARN)

        # Build graph for  training
        graph = tf.Graph()

        # Build instances of models and losses
        model_instance = model(graph=graph)
        train_loss = loss(graph=graph)
        validation_loss = loss(graph=graph)

        # Train input, loss and train_op construction
        self.logger.info(
            'Building inputs',
            extra={'phase': 'construction'})

        with graph.as_default():
            dataset_instance = dataset()
            train_input, train_label = dataset_instance.iter_train(
                batch_size=self.config.batch_size,
                epochs=self.config.epochs)

            if self.config.validate:
                validation_input, validation_label = dataset_instance.iter_train(
                    batch_size=self.config.batch_size,
                    epochs=self.config.epochs)

        # Build training part of model
        self.logger.info(
            'Building model and losses',
            extra={'phase': 'construction'})
        train_losses = train_loss.build_model_loss(
            model_instance,
            train_input,
            train_label,
            num_gpus=self.config.num_gpus,
            run_name='train')
        reg_loss = self._get_regularization_loss(model_instance)

        # Build validation part of model
        if self.config.validate:
            validation_losses = validation_loss.build_model_loss(
                model_instance,
                validation_input,
                validation_label,
                num_gpus=self.config.num_gpus,
                run_name='validation')

            total_validation_loss = (
                tf.add_n(validation_losses) /
                len(validation_losses))

        # Prepare optimizer and build train operation
        self.logger.info(
            'Building gradients and train operation',
            extra={'phase': 'construction'})
        train_outputs = self._get_train_op_multiple_gpu(
            model_instance,
            train_losses,
            reg_loss)
        train_op, total_train_loss, gradients = train_outputs
        init_op = model_instance.init_op()

        # Create summary operations
        if self.config.tensorboard_summaries:
            train_summary_op = self._build_summary_op(
                model_instance,
                train_loss,
                gradients=gradients,
                prefix='train')
            validation_summary_op = self._build_summary_op(
                model_instance,
                validation_loss,
                prefix='validation')

        # Prepare tensors for saving
        save_tensors = False
        if self.config.save_tensors:
            train_tensors = model_instance.get_tensors(
                run_name='train')
            train_tensors = {
                key: value for key, value
                in six.iteritems(train_tensors)
                if key in self.config.tensor_list
            }
            if len(train_tensors) > 0:
                save_tensors = True

        # Start session
        self.logger.info(
                'Starting session and initializing variables',
                extra={'phase': 'construction'})
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        sess = tf.Session(graph=graph, config=sess_config)
        sess.run(init_op)

        # Create tensorflow summary writers
        self.logger.info(
            'Setting up checkpoint and summary writers',
            extra={'phase': 'construction'})
        if self.config.tensorboard_summaries:
            path = os.path.join(
                self.path,
                self.config.summaries_dir)

            train_writer = tf.summary.FileWriter(
                os.path.join(path, 'train'))

            if self.config.validate:
                validation_writer = tf.summary.FileWriter(
                    os.path.join(path, 'validation'))

        # Restore model to last checkpoint
        tf_checkpoint_dir = os.path.join(
            self.path,
            self.config.tensorflow_checkpoints_dir)
        npy_checkpoint_dir = os.path.join(
            self.path,
            self.config.numpy_checkpoints_dir)

        if not os.path.exists(tf_checkpoint_dir):
            os.makedirs(tf_checkpoint_dir)

        if not os.path.exists(npy_checkpoint_dir):
            os.makedirs(npy_checkpoint_dir)

        ckpt = get_checkpoints(
            self.path,
            tf_subdir=self.config.tensorflow_checkpoints_dir,
            npy_subdir=self.config.numpy_checkpoints_dir)
        if ckpt is not None:
            ckpt_type, ckpt_path, ckpt_step = ckpt
            if ckpt_type == 'tf':
                model_instance.restore(sess, ckpt_path)
            elif ckpt_type == 'numpy':
                model_instance.numpy_restore(sess, ckpt_path)
            self.logger.info(
                'Restoring model to training step {}'.format(ckpt_step),
                extra={'phase': 'construction'})
        else:
            self.logger.info(
                'No checkpoint was found. Starting anew.',
                extra={'phase': 'construction'})

        log = (self.config.logging or self.config.tensorboard_summaries)
        checkpoints = (
            self.config.numpy_checkpoints or
            self.config.tensorflow_checkpoints)
        train_sum_freq = self.config.train_summaries_frequency
        valid_sum_freq = self.config.validation_summaries_frequency

        self.logger.info(
            'Starting training loop',
            extra={'phase': 'construction'})
        step = None
        # Main training loop
        try:
            while True:
                _, loss, step = sess.run([
                    train_op,
                    total_train_loss,
                    model_instance.global_step])

                if log:
                    if step % train_sum_freq == 0:
                        msg = '[step {st}] train_loss : {ls}'.format(
                            st=step, ls=loss)
                        self.logger.info(msg, extra={'phase': 'training'})

                        if self.config.tensorboard_summaries:
                            str_summaries = sess.run(train_summary_op)
                            train_writer.add_summary(
                                str_summaries,
                                global_step=step)

                    if self.config.validate:
                        if step % valid_sum_freq == 0:
                            loss = sess.run(total_validation_loss)
                            msg = '[step {st}] validation_loss : {ls}'
                            msg = msg.format(st=step, ls=loss)
                            self.logger.info(
                                msg,
                                extra={'phase': 'validation'})

                            if self.config.tensorboard_summaries:
                                str_summaries = sess.run(validation_summary_op)
                                validation_writer.add_summary(
                                    str_summaries,
                                    global_step=step)

                if save_tensors:
                    if step % self.config.save_tensors_frequency == 0:
                        tensor_results = sess.run(train_tensors)
                        self._save_tensors(tensor_results, step)

                if checkpoints:
                    if step % self.config.checkpoints_frequency == 0:
                        msg = '[step {st}] Checkpoint saved.'.format(st=step)
                        self.logger.info(msg, extra={'phase': 'checkpoints'})

                        if self.config.tensorflow_checkpoints:
                            model_instance.save(
                                sess,
                                os.path.join(tf_checkpoint_dir, 'ckpt'),
                                global_step=step)

                        if self.config.numpy_checkpoints:
                            model_instance.numpy_save(
                                sess,
                                npy_checkpoint_dir,
                                step=step)

        except KeyboardInterrupt:
            msg = 'User interrupted training'
            self.logger.warning(msg, extra={'phase': 'control'})

            if step is not None:
                if self.config.tensorflow_checkpoints:
                    model_instance.save(
                        sess,
                        os.path.join(tf_checkpoint_dir, 'ckpt'),
                        global_step=step)

                if self.config.numpy_checkpoints:
                    model_instance.numpy_save(
                        sess,
                        npy_checkpoint_dir,
                        step=step)

        except tf.errors.OutOfRangeError:
            msg = 'Dataset iterations done.'
            self.logging.info(msg, extra={'phase': 'control'})

            if step is not None:
                if self.config.tensorflow_checkpoints:
                    model_instance.save(
                        sess,
                        os.path.join(tf_checkpoint_dir, 'ckpt'),
                        global_step=step)

                if self.config.numpy_checkpoints:
                    model_instance.numpy_save(
                        sess,
                        npy_checkpoint_dir,
                        step=step)
