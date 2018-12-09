import os
from ..utils import parse_configs


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'default_config.yaml')


def get_config(paths=None, config=None):
    if paths is None:
        paths = []

    paths = [DEFAULT_CONFIG_PATH] + paths

    configuration = parse_configs(paths)

    if config is not None:
        configuration.update(config)

    return configuration
