import os
import configparser

from .utils import get_class
from ..evaluator.evaluator_config import get_config
from ..evaluator.evaluator import Evaluator


def evaluate(path, config, project):
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
    dataset_name = model_config['evaluation']['dataset']
    metrics_name = model_config['evaluation']['metric_list'].split(',')

    # Read classes
    model_klass = get_class(model_name, 'architecture', project, config)
    dataset_klass = get_class(dataset_name, 'dataset', project, config)

    metrics = []
    for metric in metrics_name:
        metrics.append(get_class(metric, 'metric', project, config))

    evaluator = Evaluator(eval_config, path)
    evaluator.evaluate(
        model=model_klass,
        dataset=dataset_klass,
        metrics=metrics)
