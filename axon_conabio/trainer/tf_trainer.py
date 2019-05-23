import tensorflow as tf
import numpy as np

from axon_conabio.trainer.base import Trainer


class TFTrainer(Trainer):
    def get_train_context(self):
        graph = tf.Graph()
        return {'graph': graph}

    def get_model_instance(self, model):
        graph = self.context['graph']
        model_kwargs = self.model_config['architecture']['kwargs']

        keep_prob = self.config['regularization']['keep_prob']
        model_instance = model(
            graph=graph,
            keep_prob=keep_prob,
            **model_kwargs)
        return model_instance

    def get_dataset_instance(self, dataset):
        dataset_kwargs = self.model_config['dataset']['kwargs']
        return dataset(**dataset_kwargs)

    def build_iterator_and_inputs(self, dataset_instance):
        graph = self.context['graph']
        batch_size = self.config['feed']['batch_size']
        epochs = self.config['feed']['epochs']

        with graph.as_default():
            inputs, label = dataset_instance.iter(
                batch_size=batch_size,
                epochs=epochs)

        return None, {'input': inputs, 'label': label}

    def get_loss_instance(self, loss):
        graph = self.context['graph']
        loss_kwargs = self.model_config['loss']['kwargs']

        return loss(graph=graph, **loss_kwargs)

    def build_loss(self, loss_instance, inputs, outputs):
        graph = self.context['graph']

        with graph.as_default():
            train_losses = loss_instance.build_loss(
                outputs,
                inputs['label'])

        return train_losses

    def build_outputs(self, model_instance, pipeline):
        pass

    def build_regularization_loss(self, outputs):
        pass

    def build_train_operations(self, model_instance, resources):
        pass

    def build_initialization_operations(self, model_instance):
        pass

    def intialize_training_session(self, init_operations):
        pass

    def restore_model(self, model_instance):
        pass

    def save_checkpoint(self, model_instance, step):
        pass

    def run_validation_step(self, validation_resources):
        pass

    def run_training_step(self, train_operations, train_resources, model_instance):
        pass
