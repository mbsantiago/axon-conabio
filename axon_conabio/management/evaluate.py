import os
import configparser

from .utils import load_object
from ..evaluator.evaluator_config import get_config
from ..evaluator.evaluator import Evaluator


def evaluate(path, config, project, ckpt):
    # Get training config
    evaluate_conf_file = config['configurations']['evaluator_config']
    paths = [
        os.path.join(project, '.project', evaluate_conf_file),
        os.path.join(path, evaluate_conf_file)]
    eval_config = get_config(paths=paths)

    # Read model, database and loss specifications
    model_file = config['configurations']['model_specs']
    model_config = configparser.ConfigParser()
    model_config.read([
        os.path.join(project, '.project', model_file),
        os.path.join(path, model_file)])

    model_name = model_config['model']['architecture']
    model_klass = load_object(
        model_name,
        'architecture',
        project=project,
        config=config)

    datasets_name = model_config['evaluation']['dataset'].split(',')

    if len(datasets_name) == 1 and 'metric_list' in model_config['evaluation']:
        metrics = [model_config['evaluation']['metric_list'].split(',')]

    else:
        metrics = [
            model_config['evaluation'][dataset]['metric_list']
            for dataset in datasets_name]

    evaluator = Evaluator(eval_config, path, ckpt=ckpt)
    for dataset, metric_list in zip(datasets_name, metrics):

        dataset_name = dataset.replace('_', ' ')

        try:
            name = model_config['evaluation'][dataset_name]['name']
        except:
            name = dataset

        try:
            dataset_kwargs = model_config['evaluation'][dataset_name]['dataset kwargs']
        except:
            dataset_kwargs = None

        dataset_klass = load_object(
            dataset,
            'dataset',
            project=project,
            config=config)

        metrics = [
            load_object(
                metric,
                'metric',
                project=project,
                config=config)
            for metric in metric_list]

        evaluator.evaluate(
            model=model_klass,
            dataset=dataset_klass,
            metrics=metrics,
            name=name,
            dataset_kwargs=dataset_kwargs)
