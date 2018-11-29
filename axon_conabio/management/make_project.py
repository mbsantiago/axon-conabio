import os
import shutil

from .utils import get_base_project


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def make_project(path, config):
    project = get_base_project(path)
    if project is not None:
        msg = 'Cannot create a new project within another project'
        msg += ' directory!'
        raise ValueError(msg)

    os.makedirs(path)
    struct_conf = config['structure']

    models_dir = struct_conf['models_dir']
    losses_dir = struct_conf['losses_dir']
    metrics_dir = struct_conf['metrics_dir']
    architectures_dir = struct_conf['architectures_dir']
    datasets_dir = struct_conf['datasets_dir']
    products_dir = struct_conf['products_dir']

    os.makedirs(os.path.join(path, models_dir))
    os.makedirs(os.path.join(path, losses_dir))
    os.makedirs(os.path.join(path, datasets_dir))
    os.makedirs(os.path.join(path, architectures_dir))
    os.makedirs(os.path.join(path, metrics_dir))
    os.makedirs(os.path.join(path, products_dir))
    os.makedirs(os.path.join(path, '.project'))

    shutil.copy(
        os.path.join(CURRENT_DIR, 'default_config.ini'),
        os.path.join(path, '.project', 'axon_config.ini'))
    shutil.copy(
        os.path.join(CURRENT_DIR, '../trainer/default_config.ini'),
        os.path.join(path, '.project', 'train.ini'))
