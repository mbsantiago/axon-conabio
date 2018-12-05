import os
import importlib
import sys
import configparser

from .config import get_project_config
from ..utils import get_checkpoints, memoized
from ..trainer.tf_trainer_config import get_config as get_train_config

# Classes
from ..datasets import Dataset
from ..losses import Loss
from ..metrics import Metric
from ..models import (Model, TFModel)
from ..products import Product


TYPES = {
    'dataset': {
        'class': Dataset,
        'dir': 'datasets_dir',
    },
    'loss': {
        'class': Loss,
        'dir': 'losses_dir',
    },
    'metric': {
        'class': Metric,
        'dir': 'metrics_dir',
    },
    'architecture': {
        'class': (Model, TFModel),
        'dir': 'architectures_dir',
    },
    'product': {
        'class': Product,
        'dir': 'products_dir',
    },
    'model': {
        'dir': 'models_dir',
    }
}


@memoized
def get_base_project(path):
    if not os.path.exists(path):
        return get_base_project('.')

    dirname = os.path.abspath(os.path.dirname(path))
    while dirname != '/':
        try:
            subdirs = os.listdir(dirname)
        except (IOError, OSError):
            break
        if '.project' in subdirs:
            return dirname
        dirname = os.path.dirname(dirname)
    return None


def get_all_objects(type_, project=None, config=None):
    if project is None:
        project = get_base_project('.')

    if config is None:
        # Get configuration
        config = get_project_config(project)

    subdir = TYPES[type_]['dir']
    objects_dir = config['structure'][subdir]
    objects = os.listdir(os.path.join(project, objects_dir))

    if type_ != 'model':
        objects = [
            os.path.splitext(name)[0] for name in objects
            if name[-3:] == '.py']

    return objects


def get_model_path(name, project, config):
    models_dir = config['structure']['models_dir']
    return os.path.join(project, models_dir, name)


def load_model(name=None, path=None):
    if (name is None) and (path is None):
        raise ValueError('Name or path must be supplied')

    if path is None:
        project = get_base_project('.')

    if name is None:
        name = os.path.basename(path)

    config = get_project_config(project)

    if path is None:
        path = os.path.join(
            project, config['structure']['models_dir'], name)

    model_file = config['configurations']['model_specs']
    model_config = configparser.ConfigParser()
    model_config.read([
        os.path.join(project, '.project', model_file),
        os.path.join(path, model_file)])

    architecture_name = model_config['model']['architecture']

    # To handle backwards compatibility issues
    architecture_name = architecture_name.split(':')[0]

    # Read classes
    model = load_object(
        architecture_name,
        'architecture',
        project=project,
        config=config)()

    project_train_config_path = os.path.join(
        project,
        '.project',
        config['configurations']['train_configs'])
    train_config_path = os.path.join(
        path,
        config['configurations']['train_configs'])

    train_config = get_train_config(
        paths=[project_train_config_path, train_config_path]).config

    tf_subdir = train_config['checkpoints']['tensorflow_checkpoints_dir']
    npy_subdir = train_config['checkpoints']['numpy_checkpoints_dir']
    ckpt = get_checkpoints(
        path,
        tf_subdir=tf_subdir,
        npy_subdir=npy_subdir)

    if ckpt is not None:
        ckpt_type, ckpt_path, ckpt = ckpt
        model.ckpt_type = ckpt_type
        model.ckpt_path = ckpt_path

    return model


def _extract_from_module(module, klass):
    for obj in module.__dict__.values():
        try:
            if issubclass(obj, klass) and obj is not klass:
                return obj
        except TypeError:
            pass


def load_object(name, type_, project=None, config=None):
    if project is None:
        project = get_base_project('.')

    if config is None:
        config = get_project_config(project)

    klass = TYPES[type_]['class']
    subdir_name = TYPES[type_]['dir']
    subdir = config['structure'][subdir_name]

    path = os.path.abspath(os.path.join(project, subdir))
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop()

    object_ = _extract_from_module(module, klass)
    return object_


def load_dataset(name):
    return load_object(name, 'dataset')


def load_loss(name):
    return load_object(name, 'loss')


def load_metric(name):
    return load_object(name, 'metric')


def load_architecture(name):
    return load_object(name, 'architecture')


def load_product(name):
    return load_object(name, 'product')


def get_model_checkpoint(
        model_name,
        ckpt=None):
    project = get_base_project('.')
    config = get_project_config(project)

    model_directory = os.path.join(
        project,
        config['structure']['models_dir'],
        model_name)

    if not os.path.exists(model_directory):
        msg = 'Model {} does not exists. Available models: {}'
        msg = msg.format(
            model_name,
            get_all_objects('model', config=config, project=project))
        raise IOError(msg)

    train_config = get_train_config(paths=[
        os.path.join(project, '.project', 'train.ini'),
        os.path.join(model_directory, 'train.ini')
    ])

    tf_subdir = train_config['checkpoints']['tensorflow_checkpoints_dir']
    npy_subdir = train_config['checkpoints']['numpy_checkpoints_dir']

    tf_dir = os.path.join(model_directory, tf_subdir)
    npy_dir = os.path.join(model_directory, npy_subdir)

    tf_ckpts = [
        x for x in os.listdir(tf_dir)
        if x[-6:] == '.index']

    npy_ckpts = [
        x for x in os.listdir(npy_dir)
        if x[-4:] == '.npz']

    if (not tf_ckpts) and (not npy_ckpts):
        msg = 'No checkpoints for model {} where found.'
        raise RuntimeError(msg.format(model_name))
