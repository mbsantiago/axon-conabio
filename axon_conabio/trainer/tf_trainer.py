import logging
import collections
import os
import shutil
import six
import sys
import tensorflow as tf
import numpy as np

from .tf_trainer_config import get_config, TrainConfig
from ..models.tf_model import TFModel
from ..losses.baseloss import Loss
from ..datasets.basedataset import Dataset
from ..utils import get_checkpoints


class TFTrainer(object):
    def __init__(self, config, path, retrain=False):
        if not isinstance(config, TrainConfig):
            config = get_config(config=config)

        self.config = config.config
        self.optimizer_config = config.optimizer
        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)

        if retrain:
            summaries_dir = os.path.join(
                path,
                self.config['summaries']['summaries_dir'])
            if os.path.exists(summaries_dir):
                shutil.rmtree(summaries_dir)

            npy_checkpoints_dir = os.path.join(
                path,
                self.config['checkpoints']['numpy_checkpoints_dir'])
            if os.path.exists(npy_checkpoints_dir):
                shutil.rmtree(npy_checkpoints_dir)

            tf_checkpoints_dir = os.path.join(
                path,
                self.config['checkpoints']['tensorflow_checkpoints_dir'])
            if os.path.exists(tf_checkpoints_dir):
                shutil.rmtree(tf_checkpoints_dir)

            train_log_dir = os.path.join(
                path,
                self.config['logging']['log_path'])
            if os.path.exists(train_log_dir):
                os.remove(train_log_dir)

        self._configure_logger()

    def _configure_logger(self):
        logger = logging.getLogger(__name__)

        log_config = self.config['logging']

        if not bool(log_config['logging']):
            logger.disable(logging.INFO)
            self.logger = logger
            return None

        log_format = '%(levelname)s: [%(asctime)-15s] [%(phase)s] %(message)s'
        formatter = logging.Formatter(log_format)

        verbosity = int(log_config['verbosity'])
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

        if bool(log_config['log_to_file']):
            path = os.path.join(self.path, log_config['log_path'])
            file_handler = logging.FileHandler(path)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        self.logger = logger

    def async_train(self, model=None, loss=None, dataset=None):
        # TODO
        pass

    def _get_regularization_loss(self, model):
        reg_conf = self.config['regularization']
        l1_loss = float(reg_conf['l1_loss'])
        l2_loss = float(reg_conf['l2_loss'])

        with model.graph.as_default():
            loss = 0
            if (l1_loss > 0 or l2_loss > 0):
                variables = [
                    variable for variable in model.variables.values()
                    if self._filter_regularization_variables(variable)]
                for var in variables:
                    if l1_loss > 0:
                        loss += tf.reduce_sum(tf.abs(var)) * l1_loss

                    if l2_loss > 0:
                        loss += tf.nn.l2_loss(var) * l2_loss
        return loss

    def _filter_regularization_variables(self, variable):
        if 'bias' in variable.name:
            return False
        if not variable._trainable:
            return False
        return True

    def _get_optimizer(self):
        factory, arguments = self.optimizer_config
        optimizer = factory(**arguments)
        return optimizer

    def _get_train_op_multiple_gpu(self, model, losses, reg_loss):
        architecture_conf = self.config['architecture']
        num_gpus = int(architecture_conf['num_gpus'])
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

    def _build_summary_op(
            self,
            model,
            loss,
            gradients=None,
            prefix=None,
            reg_loss=0):
        summary_conf = self.config['summaries']
        summaries = [loss.summary_op(prefix=prefix)]

        model_summs = bool(summary_conf['model_summaries'])
        var_summs = bool(summary_conf['variable_summaries'])
        grad_summs = bool(summary_conf['gradient_summaries'])
        reg_summs = bool(summary_conf['regularization_summaries'])

        if model_summs:
            model_summ_op = model.get_model_summaries(run_name=prefix)
            if model_summ_op is not None:
                summaries.append(model_summ_op)

        if var_summs:
            var_summ_op = model.get_variable_summaries(prefix=prefix)
            if model_summ_op is not None:
                summaries.append(var_summ_op)

        if (grad_summs and gradients is not None):
            grad_summ_op = self._aggregate_gradients_summaries(gradients)
            if grad_summ_op is not None:
                summaries.append(grad_summ_op)

        if reg_summs and reg_loss != 0:
            summaries.append(
                tf.summary.scalar('regularization_loss', reg_loss))

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
                summaries.append(
                    tf.summary.histogram(
                        'gradients/{name}'.format(name=name),
                        mean))

        return tf.summary.merge(summaries)

    def _save_tensors(self, tensors, step):
        tensor_conf = self.config['tensor_logs']

        directory = os.path.join(
            self.path,
            tensor_conf['tensors_dir'])
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = os.path.join(
            directory,
            'step_{}'.format(step))

        num_sample = int(tensor_conf['tensors_per_batch'])
        tensor_samples = {
            key: value[:num_sample]
            for key, value in six.iteritems(tensors)}
        np.savez(path, **tensor_samples)

    def train(
            self,
            model=None,
            loss=None,
            dataset=None,
            model_kwargs=None,
            loss_kwargs=None,
            dataset_kwargs=None):
        # Check if objects are elements of the corresponding classes
        assert issubclass(model, TFModel)
        assert issubclass(loss, Loss)
        assert issubclass(dataset, Dataset)

        if model_kwargs is None:
            model_kwargs = {}

        if loss_kwargs is None:
            loss_kwargs = {}

        if dataset_kwargs is None:
            dataset_kwargs = {}

        # Configure logging
        tf.logging.set_verbosity(tf.logging.WARN)

        # Build graph for  training
        graph = tf.Graph()

        # Build instances of models and losses
        keep_prob = float(self.config['regularization']['keep_prob'])
        model_instance = model(
            graph=graph,
            keep_prob=keep_prob,
            **model_kwargs)
        train_loss = loss(graph=graph, **loss_kwargs)
        validation_loss = loss(graph=graph, **loss_kwargs)

        # Train input, loss and train_op construction
        self.logger.info(
            'Building inputs',
            extra={'phase': 'construction'})

        validate = bool(self.config['summaries']['validate'])
        with graph.as_default():
            dataset_instance = dataset(**dataset_kwargs)
            batch_size = int(self.config['feed']['batch_size'])
            epochs = int(self.config['feed']['epochs'])
            train_input, train_label = dataset_instance.iter_train(
                batch_size=batch_size,
                epochs=epochs)

            if validate:
                validation_input, validation_label = (
                    dataset_instance.iter_validation(
                        batch_size=batch_size,
                        epochs=epochs))

        # Build training part of model
        self.logger.info(
            'Building model and losses',
            extra={'phase': 'construction'})
        num_gpus = int(self.config['architecture']['num_gpus'])
        train_losses = train_loss.build_model_loss(
            model_instance,
            train_input,
            train_label,
            num_gpus=num_gpus,
            run_name='train')
        reg_loss = self._get_regularization_loss(model_instance)

        # Build validation part of model
        if validate:
            validation_losses = validation_loss.build_model_loss(
                model_instance,
                validation_input,
                validation_label,
                num_gpus=num_gpus,
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
        tensorboard_summaries = bool(
            self.config['summaries']['tensorboard_summaries'])
        if tensorboard_summaries:
            train_summary_op = self._build_summary_op(
                model_instance,
                train_loss,
                gradients=gradients,
                reg_loss=reg_loss,
                prefix='train')
            validation_summary_op = self._build_summary_op(
                model_instance,
                validation_loss,
                prefix='validation')

        # Prepare tensors for saving
        save_tensors = bool(self.config['tensor_logs']['save_tensors'])
        if save_tensors:
            tensor_list = self.config['tensor_logs']['tensor_list'].split(',')
            train_tensors = model_instance.get_tensors(
                run_name='train')
            train_tensors = {
                key: value for key, value
                in six.iteritems(train_tensors)
                if key in tensor_list
            }
            save_tensors = len(train_tensors) > 0

        # Start session
        self.logger.info(
                'Starting session and initializing variables',
                extra={'phase': 'construction'})
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(graph=graph, config=sess_config)

        # Initialize global variables
        sess.run(init_op)
        # Intialize local variables
        with graph.as_default():
            local_init = tf.local_variables_initializer()
            tables_init = tf.tables_initializer()
        sess.run([local_init, tables_init])

        # Create tensorflow summary writers
        self.logger.info(
            'Setting up checkpoint and summary writers',
            extra={'phase': 'construction'})
        if tensorboard_summaries:
            summaries_dir = self.config['summaries']['summaries_dir']
            path = os.path.join(
                self.path,
                summaries_dir)

            train_writer = tf.summary.FileWriter(
                os.path.join(path, 'train'))

            if validate:
                validation_writer = tf.summary.FileWriter(
                    os.path.join(path, 'validation'))

        # Restore model to last checkpoint
        tf_ckp_dir = (self.config['checkpoints']
                          .get('tensorflow_checkpoints_dir'))
        npy_ckp_dir = (self.config['checkpoints']
                           .get('numpy_checkpoints_dir'))
        tf_checkpoint_dir = os.path.join(
            self.path,
            tf_ckp_dir)
        npy_checkpoint_dir = os.path.join(
            self.path,
            npy_ckp_dir)

        if not os.path.exists(tf_checkpoint_dir):
            os.makedirs(tf_checkpoint_dir)

        if not os.path.exists(npy_checkpoint_dir):
            os.makedirs(npy_checkpoint_dir)

        ckpt = get_checkpoints(
            self.path,
            tf_subdir=tf_ckp_dir,
            npy_subdir=npy_ckp_dir)
        if ckpt is not None:
            ckpt_type, ckpt_path, ckpt_step = ckpt
            model_instance.restore(sess, path=ckpt_path, mode=ckpt_type)
            self.logger.info(
                'Restoring model to training step {}'.format(ckpt_step),
                extra={'phase': 'construction'})
        else:
            self.logger.info(
                'No checkpoint was found. Starting anew.',
                extra={'phase': 'construction'})

        log = (
            bool(self.config['logging']['logging'])
            or
            tensorboard_summaries
        )
        npy_ckpts = bool(self.config['checkpoints']['numpy_checkpoints'])
        tf_ckpts = bool(self.config['checkpoints']['tensorflow_checkpoints'])
        train_sum_freq = int(
            self.config['summaries']['train_summaries_frequency'])
        valid_sum_freq = int(
            self.config['summaries']['validation_summaries_frequency'])
        save_tensors_freq = int(
            self.config['tensor_logs']['save_tensors_frequency'])
        checkpoints_freq = int(
            self.config['checkpoints']['checkpoints_frequency'])
        stop_at_step = int(self.config['feed']['stop_at_step'])
        checkpoints = (npy_ckpts or tf_ckpts)

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

                        if tensorboard_summaries:
                            str_summaries = sess.run(train_summary_op)
                            train_writer.add_summary(
                                str_summaries,
                                global_step=step)

                    if validate:
                        if step % valid_sum_freq == 0:
                            loss = sess.run(total_validation_loss)
                            msg = '[step {st}] validation_loss : {ls}'
                            msg = msg.format(st=step, ls=loss)
                            self.logger.info(
                                msg,
                                extra={'phase': 'validation'})

                            if tensorboard_summaries:
                                str_summaries = sess.run(validation_summary_op)
                                validation_writer.add_summary(
                                    str_summaries,
                                    global_step=step)

                if save_tensors:
                    if step % save_tensors_freq == 0:
                        tensor_results = sess.run(train_tensors)
                        self._save_tensors(tensor_results, step)

                if checkpoints:
                    if step % checkpoints_freq == 0:
                        msg = '[step {st}] Checkpoint saved.'.format(st=step)
                        self.logger.info(msg, extra={'phase': 'checkpoints'})

                        if tf_ckpts:
                            model_instance.save(
                                sess,
                                os.path.join(tf_checkpoint_dir, 'ckpt'),
                                global_step=step)

                        if npy_ckpts:
                            model_instance.numpy_save(
                                sess,
                                npy_checkpoint_dir,
                                step=step)

                if step == stop_at_step:
                    raise tf.errors.OutOfRangeError(None, None, "")

        except KeyboardInterrupt:
            msg = 'User interrupted training'
            self.logger.warning(msg, extra={'phase': 'control'})

            if step is not None:
                if tf_ckpts:
                    model_instance.save(
                        sess,
                        os.path.join(tf_checkpoint_dir, 'ckpt'),
                        global_step=step)

                if npy_ckpts:
                    model_instance.numpy_save(
                        sess,
                        npy_checkpoint_dir,
                        step=step)

        except tf.errors.OutOfRangeError:
            msg = 'Dataset iterations done.'
            self.logger.info(msg, extra={'phase': 'control'})

            if step is not None:
                if tf_ckpts:
                    model_instance.save(
                        sess,
                        os.path.join(tf_checkpoint_dir, 'ckpt'),
                        global_step=step)

                if npy_ckpts:
                    model_instance.numpy_save(
                        sess,
                        npy_checkpoint_dir,
                        step=step)
