import os
import configparser

from ..utils import memoized


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'default_config.ini')


@memoized
def get_config(path):
    paths = [DEFAULT_CONFIG_PATH]
    if path is not None:
        paths.append(path)

    config = configparser.ConfigParser()
    config.read(paths)

    return config


def get_project_config(project):
    project_config = os.path.join(project, '.project', 'axon_config.ini')
    return get_config(path=project_config)
