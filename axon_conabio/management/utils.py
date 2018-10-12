import os
import importlib
import sys
import click

from .config import get_config


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


def get_all_models(ctx, args, incomplete):
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
    click.echo(models_dir)
    return [x for x in os.listdir(models_dir) if incomplete in x]


def get_model_path(name, project, config):
    models_dir = config['structure']['models_dir']
    return os.path.join(project, models_dir, name)
