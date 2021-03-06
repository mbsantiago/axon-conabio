import os

import click

from .train import train as tr
from .evaluate import evaluate as ev
from .make_project import make_project as mp

from .config import get_config
from .utils import (
    get_base_project,
    get_model_path,
    get_all_models)


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
        msg = 'No model with name {name} was found. Available models: {list}'
        model_list = ', '.join(get_all_models())
        msg = msg.format(name=name, list=model_list)
        raise click.UsageError(msg)

    tr(path, config, project, retrain=retrain)


@main.command()
@click.argument('type', type=click.Choice([
    'architectures',
    'losses',
    'metrics',
    'models',
    'datasets']))
@click.option('--path')
def list(type, path):
    if path is None:
        path = '.'

    project = get_base_project(path)
    config_path = os.path.join(
            project, '.project', 'axon_config.ini')
    config = get_config(path=config_path)
    structure = config['structure']

    if type == 'models':
        result = os.listdir(
            os.path.join(
                project,
                structure['models_dir']))
    else:
        type_name = type + '_dir'
        directory = os.path.join(
            project,
            structure[type_name])
        result = [
            os.path.splitext(os.path.basename(x))[0]
            for x in os.listdir(directory)
            if x[-3:] == '.py']

    msg = 'Available {}:'.format(type)
    for n, name in enumerate(result):
        msg += '\n\t{}. {}'.format(n + 1, name)

    click.echo(msg)


@main.command()
@click.argument('name', required=False)
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
        msg = 'No model with name {name} was found. Available models: {list}'
        model_list = ', '.join(get_all_models())
        msg = msg.format(name=name, list=model_list)
        raise click.UsageError(msg)

    ev(path, config, project)


@main.command()
@click.argument('path', type=click.Path(exists=False))
@click.option('--config', type=click.Path(exists=True))
def make_project(path, config):
    config = get_config(path=config)
    mp(path, config)
