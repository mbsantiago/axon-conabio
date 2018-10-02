import os
import configparser

from .utils import get_class
from ..trainer.tf_trainer_config import get_config
from ..trainer.tf_trainer import TFTrainer


def train(path, config, project):
    # Get training config
    train_config_file = config['configurations']['train_configs']
    paths = [
        os.path.join(project, '.project', train_config_file),
        os.path.join(path, train_config_file)]

    # Read model, database and loss specifications
    model_file = config['configurations']['model_specs']
    model_config = configparser.ConfigParser()
    model_config.read([
        os.path.join(project, '.project', model_file),
        os.path.join(path, model_file)])

    architecture_name = model_config['model']['architecture']
    dataset_name = model_config['training']['dataset']
    loss_name = model_config['training']['loss']

    # Read classes
    model_klass = get_class(architecture_name, 'architecture', project, config)
    dataset_klass = get_class(dataset_name, 'dataset', project, config)
    loss_klass = get_class(loss_name, 'loss', project, config)

    train_config = get_config(paths=paths)
    trainer = TFTrainer(train_config, path)
    trainer.train(model=model_klass, dataset=dataset_klass, loss=loss_klass)
