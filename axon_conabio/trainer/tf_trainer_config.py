import os
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
        'variable_summaries': False,
        'gradient_summaries': True,
        'train_frequency': 50,
        'validation_frequency': 200,
    },
    'regularization': {
        'l1_loss': 0.0,
        'l2_loss': 0.0,
        'keep_prob': 0.5,
    },
    'architecture': {
        'distributed': False,
        'num_gpus': 2,
    },
    'batch_size': 100,
    'epochs': 10000,
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

        # Summary options
        user_options = config.get('summaries', {})
        default_conf = TRAINER_CONFIG_FIELDS['summaries']

        variable_summaries = user_options.get(
            'variable_summaries',
            default_conf['variable_summaries'])
        setattr(self, 'variable_summaries', variable_summaries)

        gradient_summaries = user_options.get(
            'gradient_summaries',
            default_conf['gradient_summaries'])
        setattr(self, 'gradient_summaries', gradient_summaries)

        summaries_every = user_options.get(
            'train_frequency',
            default_conf['train_frequency'])
        setattr(self, 'train_summaries_frequency', summaries_every)

        summaries_every = user_options.get(
            'validation_frequency',
            default_conf['validation_frequency'])
        setattr(self, 'validation_summaries_frequency', summaries_every)

        # Regularization options
        user_options = config.get('regularization', {})
        default_conf = TRAINER_CONFIG_FIELDS['regularization']

        keep_prob = user_options.get(
            'keep_prob',
            default_conf['keep_prob'])
        setattr(self, 'keep_prob', keep_prob)

        l1_loss = user_options.get(
            'l1_loss',
            default_conf['l1_loss'])
        setattr(self, 'l1_loss', l1_loss)

        l2_loss = user_options.get(
            'l2_loss',
            default_conf['l2_loss'])
        setattr(self, 'l2_loss', l2_loss)

        # Architecture options
        user_options = config.get('architecture', {})
        default_conf = TRAINER_CONFIG_FIELDS['architecture']

        num_gpus = user_options.get(
            'num_gpus',
            default_conf['num_gpus'])
        setattr(self, 'num_gpus', num_gpus)

        distributed = user_options.get(
            'distributed',
            default_conf['distributed'])
        setattr(self, 'distributed', distributed)

        # Feed options
        num_epochs = config.get(
            'epochs',
            TRAINER_CONFIG_FIELDS['epochs'])
        setattr(self, 'epochs', num_epochs)

        batch_size = config.get(
            'batch_size',
            TRAINER_CONFIG_FIELDS['batch_size'])
        setattr(self, 'batch_size', batch_size)

    def __str__(self):
        msg = 'Train configuration:' + '\n'
        msg += '  - Optimizer:' + '\n'
        msg += '    name : ' + self.optimizer + '\n'
        msg += '    arguments : ' + str(self.optimizer_arguments) + '\n'
        msg += '  - Regularization:' + '\n'
        msg += '    l1 : ' + str(self.l1_loss) + '\n'
        msg += '    l2 : ' + str(self.l2_loss) + '\n'
        msg += '    keep_prob : ' + str(self.keep_prob) + '\n'
        msg += '  - Summaries:' + '\n'
        msg += '    variable_summaries : ' + str(self.variable_summaries) + '\n'
        msg += '    gradient_summaries : ' + str(self.gradient_summaries) + '\n'
        msg += '    train_summaries_freq : ' + str(self.train_summaries_frequency) + '\n'
        msg += '    valid_summaries_freq : ' + str(self.validation_summaries_frequency) + '\n'
        msg += '  - Architecture:' + '\n'
        msg += '    distributed : ' + str(self.distributed) + '\n'
        msg += '    num_gpus : ' + str(self.num_gpus) + '\n'
        msg += '  - Feed:' + '\n'
        msg += '    batch_size : ' + str(self.batch_size) + '\n'
        msg += '    epochs : ' + str(self.epochs)
        return msg

    def __repr__(self):
        msg = 'TrainConfig({d})'.format(d=self.dict)
        return msg
