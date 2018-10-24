import os

import click

from .train import train as tr
from .evaluate import evaluate as ev
from .make_project import make_project as mp

from .config import get_config
from .utils import get_base_project, get_model_path, get_all_models


@click.group()
def main():
    pass


@main.command()
@click.argument('name', required=False, autocompletion=get_all_models)
@click.option('--path')
def train(name, path):
    # Get current project
    if name is not None:
        project = get_base_project('.')
    elif path is not None:
        project = get_base_project(path)
    else:
        msg = 'Name of model or path to model must be supplied'
        raise click.UsageError(msg)

    # Get configuration
    config_path = None
    if project is not None:
        config_path = os.path.join(
                project, '.project', 'axon_config.ini')
    config = get_config(path=config_path)

    # If name was given
    if name is not None:
        path = get_model_path(name, project, config)

    if not os.path.exists(path):
        msg = 'No model with name {} was found'.format(name)
        raise click.UsageError(msg)

    tr(path, config, project)


@main.command()
@click.argument('name', required=False, autocompletion=get_all_models)
@click.option('--path')
def evaluate(name, path):
    # Get current project
    if name is not None:
        project = get_base_project('.')
    elif path is not None:
        project = get_base_project(path)
    else:
        msg = 'Name of model or path to model must be supplied'
        raise click.UsageError(msg)

    # Get configuration
    config_path = None
    if project is not None:
        config_path = os.path.join(
                project, '.project', 'axon_config.ini')
        config = get_config(path=config_path)

    # If name was given
    if name is not None:
        path = get_model_path(name, project, config)

    if not os.path.exists(path):
        msg = 'No model with name {} was found'.format(name)
        raise click.UsageError(msg)

    ev(path, config, project)


@main.command()
@click.argument('path', type=click.Path(exists=False))
@click.option('--config', type=click.Path(exists=True))
def make_project(path, config):
    config = get_config(path=config)
    mp(path, config)
