import os
import six
import json
import yaml


EVALUATOR_CONFIG_FIELDS = {
    'logging': {
        'logging': True,
        'log_to_file': False,
        'log_path': 'evaluation.log',
        'verbosity': 3,
    },
    'evaluations': {
        'evaluations_dir': 'evaluations',
        'save_predictions': False,
        'predictions_dir': 'evaluations/predictions',
        'save_results': True,
        'results_format': 'csv',
    },
    'other': {
        'checkpoints_dir': 'checkpoints'
    }
}


class ConfigError(Exception):
    pass


class EvaluatorConfig(object):
    def __init__(self, config):
        self.dict = config
        self._parse_config(config)

    @classmethod
    def from_path(cls, path, name='train_config'):
        config_file = os.path.join(path, name)

        json_exists = os.path.exists(config_file + '.json')
        yaml_exists = os.path.exists(config_file + '.yaml')

        if (not json_exists and not yaml_exists):
            msg = 'No configuration file for training was found at {path}'
            msg = msg.format(path=path)
            raise IOError(msg)

        if json_exists:
            with open(config_file + '.json', 'r') as jfile:
                config = json.load(jfile)
        else:
            with open(config_file + '.yaml', 'r') as yfile:
                config = yaml.load(yfile)

        return cls(config)

    def _parse_config(self, config):
        # For other configurations
        for key, value in six.iteritems(EVALUATOR_CONFIG_FIELDS):
            default_config = EVALUATOR_CONFIG_FIELDS[key]
            user_config = config.get(key, {})

            for subkey, subvalue in six.iteritems(default_config):
                attr_value = user_config.get(subkey, subvalue)
                setattr(self, subkey, attr_value)

    def __str__(self):
        msg = 'Evaluator configuration:' + '\n'

        for key, value in six.iteritems(EVALUATOR_CONFIG_FIELDS):
            if key == 'optimizer':
                continue

            msg += '  - {theme}:\n'.format(theme=key)
            for subkey in value:
                conf_value = getattr(self, subkey)
                new_msg = '      {key} : {value}\n'
                new_msg = new_msg.format(key=subkey, value=conf_value)
                msg += new_msg

        return msg

    def __repr__(self):
        msg = 'EvaluatorConfig({d})'.format(d=self.dict)
        return msg
