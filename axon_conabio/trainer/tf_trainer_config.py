import os
import configparser

import tensorflow as tf


DEFAULT_CONFIG_PATH = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    'default_config.ini'
)

OPTIMIZERS = {
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


class TrainConfig(object):
    def __init__(self, config):
        self.config = config
        self.optimizer = get_optimizer(config)


def get_optimizer(config):
    optimizer = config['optimizer']['name']
    factory = OPTIMIZERS[optimizer]['factory']
    arguments = OPTIMIZERS[optimizer]['arguments'].copy()

    for key in arguments:
        if key in config['optimizer']:
            dtype = type(arguments[key])
            arguments[key] = dtype(config['optimizer'][key])

    return (factory, arguments)


def get_config(paths=None, config=None):
    if paths is None:
        paths = []

    train_config = configparser.ConfigParser()
    train_config.read([DEFAULT_CONFIG_PATH] + paths)

    if config is not None:
        train_config.read_dict(config)

    return TrainConfig(train_config)
