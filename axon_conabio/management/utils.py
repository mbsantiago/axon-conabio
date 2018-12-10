import os
import importlib
import sys
from contextlib import contextmanager

from .config import get_project_config
from ..utils import get_checkpoints, memoized, parse_configs
from ..trainer.tf_trainer_config import get_config as get_train_config

# Classes
from ..datasets import Dataset
from ..losses import Loss
from ..metrics import (Metric, MetricBundle, ThresholdedMetric)
from ..models import (Model, TFModel)
from ..products import Product
from ..preprocessors import Preprocessor


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
        'class': (Metric, MetricBundle, ThresholdedMetric),
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
    },
    'preprocessor': {
        'class': Preprocessor,
        'dir': 'preprocessors_dir',
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
        project = get_base_project(os.path.abspath('.') + '/')

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


def load_model(name=None, path=None, ckpt=None):
    if (name is None) and (path is None):
        raise ValueError('Name or path must be supplied')

    if path is None:
        project = get_base_project(os.path.abspath('.') + '/')
    else:
        project = get_base_project(path)

    if name is None:
        name = os.path.basename(path)

    config = get_project_config(project)

    if path is None:
        path = os.path.join(
            project, config['structure']['models_dir'], name)

    model_file = config['configurations']['model_specs']

    if not os.path.exists(os.path.join(path, model_file)):
        msg = 'Model {} does not exists. Available models: {}'
        msg = msg.format(
            os.path.basename(path),
            ' '.join(list_models()))
        raise IOError(msg)

    model_config = parse_configs([
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

    try:
        ckpt_type, ckpt_path, _ = get_model_checkpoint(
            name,
            ckpt=ckpt)

        model.ckpt_type = ckpt_type
        model.ckpt_path = ckpt_path

    except RuntimeError:
        pass

    return model


def _extract_from_module(module, klass):
    if not isinstance(klass, (tuple, list)):
        klass = (klass, )

    for obj in module.__dict__.values():
        try:
            if issubclass(obj, klass) and obj not in list(klass):
                return obj
        except TypeError:
            pass


@contextmanager
def dir_in_path(dirpath):
    sys.path.insert(0, dirpath)
    yield
    sys.path.pop(0)


def load_object(name, type_, project=None, config=None):
    # Backwards compatibility
    name = name.split(':')[0]

    if project is None:
        project = get_base_project(os.path.abspath('.') + '/')

    if config is None:
        config = get_project_config(project)

    klass = TYPES[type_]['class']
    subdir_name = TYPES[type_]['dir']
    subdir = config['structure'][subdir_name]

    path = os.path.abspath(os.path.join(project, subdir))
    with dir_in_path(path):
        module = importlib.import_module(name)

    object_ = _extract_from_module(module, klass)
    return object_


def load_dataset(name):
    return load_object(name, 'dataset')


def list_datasets():
    return get_all_objects('dataset')


def load_loss(name):
    return load_object(name, 'loss')


def list_losses():
    return get_all_objects('loss')


def load_metric(name):
    return load_object(name, 'metric')


def list_metrics():
    return get_all_objects('metric')


def load_architecture(name):
    return load_object(name, 'architecture')


def list_architectures():
    return get_all_objects('architecture')


def load_product(name):
    return load_object(name, 'product')


def list_products():
    return get_all_objects('product')


def load_preprocessor(name):
    return load_object(name, 'preprocessor')


def list_preprocessors():
    return get_all_objects('preprocessor')


def list_models():
    return get_all_objects('model')


def get_model_checkpoint(
        model_name,
        ckpt=None):
    project = get_base_project(os.path.abspath('.') + '/')
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
        os.path.join(project, '.project', 'train.yaml'),
        os.path.join(model_directory, 'train.yaml')
    ])

    tf_subdir = train_config.config['checkpoints']['tensorflow_checkpoints_dir']
    npy_subdir = train_config.config['checkpoints']['numpy_checkpoints_dir']

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

    if ckpt is None:
        ckpt = 0
        index = -1
    else:
        index = 0

    # Order checkpoints by closeness to desired checkpoint number
    tf_ckpts = sorted(
        tf_ckpts,
        key=lambda x: abs(ckpt - int(x.split('.')[0].split('-')[1])))

    npy_ckpts = sorted(
        npy_ckpts,
        key=lambda x: abs(ckpt - int(x.split('.')[0].split('_')[2])))

    tf_ckpt = tf_ckpts[index] if tf_ckpts else None
    npy_ckpt = npy_ckpts[index] if npy_ckpts else None

    if tf_ckpt is None:
        ckpt_type = 'npy'
        ckpt_name = npy_ckpt
        ckpt_step = int(npy_ckpt.split('.')[0].split('_')[2])

    elif npy_ckpt is None:
        ckpt_type = 'tf'
        ckpt_name = tf_ckpt
        ckpt_step = int(tf_ckpt.split('.')[0].split('-')[1])

    else:
        tf_ckpt_nmbr = int(tf_ckpt.split('.')[0].split('-')[1])
        npy_ckpt_nmbr = int(npy_ckpt.split('.')[0].split('_')[2])

        if index == -1:
            if tf_ckpt_nmbr > npy_ckpt_nmbr:
                ckpt_type = 'tf'
                ckpt_name = tf_ckpt
                ckpt_step = tf_ckpt_nmbr
            else:
                ckpt_type = 'npy'
                ckpt_name = npy_ckpt
                ckpt_step = npy_ckpt_nmbr

        else:
            if abs(ckpt - tf_ckpt_nmbr) < abs(ckpt - npy_ckpt_nmbr):
                ckpt_type = 'tf'
                ckpt_name = tf_ckpt
                ckpt_step = tf_ckpt_nmbr
            else:
                ckpt_type = 'npy'
                ckpt_name = npy_ckpt
                ckpt_step = npy_ckpt_nmbr

    if ckpt_type == 'npy':
        subdir = npy_subdir
    else:
        subdir = tf_subdir

    ckpt_path = os.path.join(model_directory, subdir, ckpt_name)
    return ckpt_type, ckpt_path, ckpt_step
