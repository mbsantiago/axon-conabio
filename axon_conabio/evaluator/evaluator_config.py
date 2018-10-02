import os
import configparser


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'default_config.ini')


def get_config(paths=None, config=None):
    if paths is None:
        paths = []

    evaluator_config = configparser.ConfigParser()
    evaluator_config.read([DEFAULT_CONFIG_PATH] + paths)

    if config is not None:
        evaluator_config.read_dict(config)

    return evaluator_config
