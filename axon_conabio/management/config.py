import os
import configparser


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'default_config.ini')


def get_config(path=None):
    paths = [DEFAULT_CONFIG_PATH]
    if path is not None:
        paths.append(path)

    config = configparser.ConfigParser()
    config.read([DEFAULT_CONFIG_PATH] + paths)

    return config
