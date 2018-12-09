import os

import click

from .train import train as tr
from .evaluate import evaluate as ev
from .make_project import make_project as mp

from .config import get_config
from .utils import (
    get_base_project,
    get_model_path,
    get_all_objects)


@click.group()
def main():
    pass


@main.command()
@click.argument('name', required=False)
@click.option('--path')
@click.option('--retrain', is_flag=True)
def train(name, path, retrain):

    if retrain:
        msg = 'Retrain option is set. This will erase all summaries'
        msg += ' and checkpoints currently available for this model.'
        msg += ' Do you wish to continue?'
        click.confirm(msg, abort=True)

    # Get current project
    if name is not None:
        project = get_base_project(os.path.abspath('.') + '/')
    elif path is not None:
        project = get_base_project(path)
    else:
        msg = 'Name of model or path to model must be supplied'
        raise click.UsageError(msg)

    # Get configuration
    config_path = None
    if project is not None:
        config_path = os.path.join(
                project, '.project', 'axon_config.yaml')
    config = get_config(path=config_path)

    # If name was given
    if name is not None:
        path = get_model_path(name, project, config)

    if not os.path.exists(path):
        msg = 'No model with name {name} was found. Available models: {list}'
        model_list = ', '.join(get_all_objects('model'))
        msg = msg.format(name=name, list=model_list)
        raise click.UsageError(msg)

    tr(path, config, project, retrain=retrain)


@main.command()
@click.argument('type', type=click.Choice([
    'architecture',
    'loss',
    'metric',
    'model',
    'product',
    'dataset']))
@click.option('--path')
def list(type, path):
    if path is None:
        path = '.'

    project = get_base_project(path)
    config_path = os.path.join(
            project, '.project', 'axon_config.yaml')
    config = get_config(path=config_path)

    result = get_all_objects(
        type,
        project=project,
        config=config)

    msg = 'Available {}:'.format(type)
    for n, name in enumerate(result):
        msg += '\n\t{}. {}'.format(n + 1, name)

    click.echo(msg)


@main.command()
@click.argument('name', required=False)
@click.option('--path')
@click.option('--ckpt', type=int)
def evaluate(name, path, ckpt):
    # Get current project
    if name is not None:
        project = get_base_project(os.path.abspath('.') + '/')
    elif path is not None:
        project = get_base_project(path)
    else:
        msg = 'Name of model or path to model must be supplied'
        raise click.UsageError(msg)

    if project is None:
        msg = 'You (or the target directory) are not inside an'
        msg += ' axon project!'
        raise click.UsageError(msg)

    # Get configuration
    config_path = os.path.join(
        project, '.project', 'axon_config.yaml')
    config = get_config(path=config_path)

    # If name was given
    if name is not None:
        path = get_model_path(name, project, config)

    if not os.path.exists(path):
        msg = 'No model with name {name} was found. Available models: {list}'
        model_list = ', '.join(
            get_all_objects('model', project=project, config=config))
        msg = msg.format(name=name, list=model_list)
        raise click.UsageError(msg)

    ev(path, config, project, ckpt)


@main.command()
@click.argument('path', type=click.Path(exists=False))
@click.option('--config', type=click.Path(exists=True))
def make_project(path, config):
    config = get_config(path=config)
    mp(path, config)
