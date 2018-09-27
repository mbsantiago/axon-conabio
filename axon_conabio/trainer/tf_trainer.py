import tensorflow as tf

from .tf_trainer_config import TrainerConfig
from ..models.tf_model import TFModel
from ..loss.baseloss import Loss
from ..datasets.basedataset import Dataset


class TFTrainer(object):
    def __init__(self, config):
        if not isinstance(config, TrainerConfig):
            config = TrainerConfig(config)
        self.config = config

    def async_train(self, model=None, loss=None, dataset=None):
        # TODO
        pass

    def _get_regularization_loss(self, model):
        with model.graph.as_default():
            with tf.name_scope('loss/regularization'):
                loss = 0
                if (self.config.l1_loss > 0 or self.config.l2_loss > 0):
                    variables = [
                        variable for variable in model.variables.values()
                        if self.filter_regularization_variables(variable)]
                    for var in variables:
                        if self.config.l1_loss > 0:
                            loss += tf.reduce_sum(tf.abs(var))

                        if self.config.l2_loss > 0:
                            loss += tf.nn.l2_loss(var)
        return loss

    def filter_regularization_variables(self, variable):
        if 'bias' in variable.name:
            return False
        if not variable._trainable:
            return False
        return True

    def _get_optimizer(self):
        with tf.name_scope('optimizer'):
            arguments = self.config.optimizer_arguments
            factory = self.config.optimizer_factory
            optimizer = factory(**arguments)
        return optimizer

    def _get_train_op(self, model, losses, reg_loss):
        with model.graph.as_default():
            with tf.name_scope('training/'):
                with tf.name_scope('total_loss'):
                    total_loss = tf.add_n(losses) / len(losses) + reg_loss
                optimizer = self._get_optimizer()
                train_op = optimizer.minimize(
                    total_loss,
                    global_step=model.global_step)
            return train_op, total_loss

    def _get_train_op_multiple_gpu(self, model, losses, reg_loss):
        num_gpus = self.config.num_gpus

        if num_gpus == 1:
            return self._get_train_op(model, losses, reg_loss)

        assert len(losses) == num_gpus

        with model.graph.as_default():
            with tf.name_scope('training/'):
                optimizer = self._get_optimizer()
                gradients_list = []

                for i in range(num_gpus):
                    with tf.device('/device:GPU:{}'.format(i)):
                        scope_name = 'tower_{}_gradients'.format(i)
                        with tf.name_scope(scope_name):
                            gradients = optimizer.compute_gradients(
                                losses[i] / num_gpus)
                            gradients_list += gradients

                if reg_loss != 0:
                    with tf.name_scope('regularization'):
                        gradients_list += optimizer.compute_gradients(reg_loss)

                train_op = optimizer.apply_gradients(gradients_list)

                with tf.name_scope('total_loss'):
                    total_loss = tf.add_n(losses) / num_gpus + reg_loss

        return train_op, total_loss

    def train(self, model=None, loss=None, dataset=None):
        # Check if objects are elements of the corresponding classes
        assert isinstance(model, TFModel)
        assert isinstance(loss, Loss)
        assert isinstance(dataset, Dataset)

        # Train input, loss and train_op construction
        with model.graph.as_default():
            train_input, train_label = dataset.iter_train(
                batch_size=self.config.batch_size,
                epochs=self.config.epochs)
            validation_input, validation_label = dataset.iter_train(
                batch_size=self.config.batch_size,
                epochs=self.config.epochs)

        losses = loss.build_model_loss(
            model,
            train_input,
            train_label,
            num_gpus=self.config.num_gpus)
        reg_loss = self._get_regularization_loss(model)

        train_op, train_loss = self._get_train_op_multiple_gpus(
            model,
            losses,
            reg_loss)

        init_op = model.init_op()

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True)
        sess = tf.Session(graph=model.graph, config=sess_config)

        sess.run(init_op)

        while True:
            _, loss, step = sess.run([train_op, train_loss, model.global_step])

            if step % self.config.train_summaries_frequency == 0:
                msg = '[STEP: {st}] Loss: {ls}'.format(st=step, ls=loss)
                print(msg)
