import os
import configparser

from .utils import load_object
from ..trainer.tf_trainer_config import get_config
from ..trainer.tf_trainer import TFTrainer


def train(path, config, project, retrain=False):
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

    architecture_name = model_config['model']['architecture'].split(':')[0]
    dataset_name = model_config['training']['dataset'].split(':')[0]
    loss_name = model_config['training']['loss'].split(':')[0]

    model_kwargs = model_config['model'].get('model_kwargs', None)
    dataset_kwargs = model_config['training'].get('dataset_kwargs', None)
    loss_kwargs = model_config['training'].get('loss_kwargs', None)

    # Read classes
    model_klass = load_object(
        architecture_name,
        'architecture',
        project=project,
        config=config)

    dataset_klass = load_object(
        dataset_name,
        'dataset',
        project=project,
        config=config)

    loss_klass = load_object(
        loss_name,
        'loss',
        project=project,
        config=config)

    train_config = get_config(paths=paths)
    trainer = TFTrainer(train_config, path, retrain=retrain)
    trainer.train(
        model=model_klass,
        dataset=dataset_klass,
        loss=loss_klass,
        model_kwargs=model_kwargs,
        dataset_kwargs=dataset_kwargs,
        loss_kwargs=loss_kwargs)
