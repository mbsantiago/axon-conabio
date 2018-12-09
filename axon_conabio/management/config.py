import os

from ..utils import memoized, parse_configs


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'default_config.yaml')


@memoized
def get_config(path=None, config=None):
    if path is None:
        path = []
    else:
        path = [path]

    paths = [DEFAULT_CONFIG_PATH] + path

    configuration = parse_configs(paths)

    if config is not None:
        configuration.update(config)

    return configuration


def get_project_config(project):
    project_config = os.path.join(project, '.project', 'axon_config.yaml')
    return get_config(path=project_config)
