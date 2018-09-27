import os
import six
import json
import yaml

import tensorflow as tf


TRAINER_CONFIG_FIELDS = {
    'optimizer': {
        'default': 'Adam',
        'options': {
            'GradientDescent': {
                'factory': tf.train.GradientDescentOptimizer,
                'arguments': {
                    'learning_rate': 0.001
                },
            },
            'Adadelta': {
                'factory': tf.train.AdadeltaOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'rho': 0.95,
                    'epsilon': 1e-08,
                },
            },
            'Adagrad': {
                'factory': tf.train.AdagradOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'initial_accumulator_value': 0.1,
                }
            },
            'AdagradDA': {
                'factory': tf.train.AdagradDAOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'global_step': None,
                    'initial_gradient_squared_accumulator_value': 0.1,
                    'l1_regularization_strength': 0.0,
                    'l1_regularization_strength': 0.0,
                },
            },
            'Momentum': {
                'factory': tf.train.MomentumOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'momentum': 0.999,
                    'use_nesterov': False,
                },
            },
            'Adam': {
                'factory': tf.train.AdamOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'beta1': 0.9,
                    'beta2': 0.999,
                    'epsilon': 1e-08,
                },
            },
            'Ftrl': {
                'factory': tf.train.FtrlOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'learning_rate_power': -0.5,
                    'initial_accumulator_value': 0.1,
                    'l1_regularization_strength': 0.0,
                    'l2_regularization_strength': 0.0,
                    'l2_shrinkage_regularization_strength': 0.0,
                },
            },
            'ProximalGradientDescent': {
                'factory': tf.train.ProximalGradientDescentOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'l1_regularization_strength': 0.0,
                    'l2_regularization_strength': 0.0,
                },
            },
            'ProximalAdagrad': {
                'factory': tf.train.ProximalAdagradOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'initial_accumulator_value': 0.1,
                    'l1_regularization_strength': 0.0,
                    'l2_regularization_strength': 0.0,
                },
            },
            'RMSProp': {
                'factory': tf.train.RMSPropOptimizer,
                'arguments': {
                    'learning_rate': 0.001,
                    'decay': 0.9,
                    'momentum': 0.0,
                    'epsilon': 1e-10,
                    'centered': True,
                }
            }
        }
    },
    'summaries': {
        'validate': True,
        'tensorboard_summaries': True,
        'variable_summaries': False,
        'gradient_summaries': True,
        'custom_summaries': True,
        'save_graph': True,
        'train_summaries_frequency': 50,
        'validation_summaries_frequency': 200,
        'custom_summaries_dir': 'custom_summaries',
        'summaries_dir': 'summaries',
    },
    'logging': {
        'logging': True,
        'log_to_file': False,
        'log_path': 'training.log',
        'verbosity': 3,
    },
    'checkpoints': {
        'checkpoints_frequency': 200,
        'checkpoints_dir': 'checkpoints',
        'numpy_checkpoints': False,
    },
    'regularization': {
        'l1_loss': 0.0,
        'l2_loss': 0.0,
        'keep_prob': 0.5,
    },
    'architecture': {
        'distributed': False,
        'distributed_config': 'cluster_spec.py',
        'num_gpus': 1,
    },
    'feed': {
        'batch_size': 100,
        'epochs': 10000,
    }
}


class ConfigError(Exception):
    pass


class TrainerConfig(object):
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
        # Select optimizer and its arguments
        optimizer_conf = TRAINER_CONFIG_FIELDS['optimizer']
        if 'optimizer' in config:
            try:
                name = config['optimizer']['name']
            except KeyError:
                name = optimizer_conf['default']

            if name not in optimizer_conf['options']:
                msg = 'Optimizer {name} is no implemented. Options: {options}'
                msg = msg.format(
                        name=name,
                        options=str(optimizer_conf['options'].keys()))
                raise ConfigError(msg)
        else:
            name = optimizer_conf['default']

        user_confs = config.get('optimizer', {})
        optimizer_conf = optimizer_conf['options'][name]

        # Set optimizer name
        setattr(self, 'optimizer', name)
        setattr(self, 'optimizer_factory', optimizer_conf['factory'])

        # Build optimizer build arguments
        arguments = optimizer_conf['arguments'].copy()
        for key in user_confs:
            if key in arguments:
                arguments[key] = user_confs[key]
        setattr(self, 'optimizer_arguments', arguments)

        # For other configurations
        for key, value in six.iteritems(TRAINER_CONFIG_FIELDS):
            if key == 'optimizer':
                continue

            default_config = TRAINER_CONFIG_FIELDS[key]
            user_config = config.get(key, {})

            for subkey, subvalue in six.iteritems(default_config):
                attr_value = user_config.get(subkey, subvalue)
                setattr(self, subkey, attr_value)

    def __str__(self):
        msg = 'Train configuration:' + '\n'
        msg += '  - optimizer:\n'
        msg += '      name : {value}\n'.format(value=self.optimizer)
        for key, value in six.iteritems(self.optimizer_arguments):
            msg += '      {key} : {value}\n'.format(key=key, value=value)

        for key, value in six.iteritems(TRAINER_CONFIG_FIELDS):
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
        msg = 'TrainConfig({d})'.format(d=self.dict)
        return msg
