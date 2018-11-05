import os
import importlib
import sys
import logging
import configparser

from .config import get_config
from ..utils import get_checkpoints
from ..trainer.tf_trainer_config import get_config as get_train_config


logger = logging.getLogger(__name__)


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


def get_class(name, subtype, project, config):
    # Get class name
    split = name.split(':')
    if len(split) == 1:
        name = split[0]
        klass_name = subtype
    else:
        name = split[0]
        klass_name = split[1]

    # Check if name is path to file
    abspath = os.path.abspath(name)
    if os.path.exists(abspath):
        name, ext = os.path.splitext(abspath)
        if ext != '.py':
            msg = '{} is being loaded from a non-python file: {}'
            msg = msg.format(subtype.title(), abspath)
            raise IOError(msg)

        klass = extract_class(name, klass_name)
    else:
        if project is None:
            msg = 'No path to {subtype} file :{file}: was given'
            msg += ' and training is not being done inside'
            msg += ' a project.'
            msg = msg.format(subtype=subtype, file=name)
            raise IOError(msg)

        if subtype == 'architecture':
            subdir = config['structure']['architectures_dir']
        elif subtype == 'loss':
            subdir = config['structure']['losses_dir']
        elif subtype == 'dataset':
            subdir = config['structure']['datasets_dir']
        elif subtype == 'metric':
            subdir = config['structure']['metrics_dir']

        path = os.path.join(project, subdir, name)
        klass = extract_class(path, klass_name)

    return klass


def extract_class(path, name):
    basename = os.path.basename(path)
    abspath = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, abspath)
    module = importlib.import_module(basename)
    try:
        klass = getattr(module, name)
    except AttributeError:
        msg = 'Python module {} does not have a class named {}'
        msg = msg.format(os.path.basename(path), name)
        raise AttributeError(msg)
    return klass


def get_all_models():
    project = get_base_project('.')

    # Get configuration
    config_path = None
    if project is not None:
        config_path = os.path.join(
            project, '.project', 'axon_config.ini')
    else:
        return []
    config = get_config(path=config_path)
    models_dir = config['structure']['models_dir']
    return os.listdir(models_dir)


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

    config_path = os.path.join(
        project, '.project', 'axon_config.ini')
    config = get_config(path=config_path)

    if path is None:
        path = os.path.join(
            project, config['structure']['models_dir'], name)

    model_file = config['configurations']['model_specs']
    model_config = configparser.ConfigParser()
    model_config.read([
        os.path.join(project, '.project', model_file),
        os.path.join(path, model_file)])

    architecture_name = model_config['model']['architecture']

    # Read classes
    model = get_class(
        architecture_name,
        'architecture',
        project,
        config)()

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
