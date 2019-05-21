import logging
import os

from axon_conabio.exceptions import ConfigError


LOG_FORMAT = '%(levelname)s: [%(asctime)-15s] [%(phase)s] %(message)s'
VERBOSITY_LEVELS = {
    1: logging.ERROR,
    2: logging.WARNING,
    3: logging.INFO,
    4: logging.DEBUG
}


class BaseLogger(object):
    def __init__(self, name, config, path=None):
        self.name = name
        self.config = config
        self.path = path

        self.check_config()

    def get_logger(self):
        logger = logging.getLogger(self.name)
        self.configure_logger(logger)
        return logger

    def configure_logger(self, logger):
        log_config = self.config

        if not bool(log_config['logging']):
            logger.disable(logging.INFO)
            return None

        console_handler = logging.StreamHandler(sys.stdout)

        self.set_logger_formatter(console_handler)
        self.set_logger_level(console_handler)

        logger.addHandler(console_handler)

        if bool(log_config['log_to_file']):
            self.add_file_handler(logger)

    def add_file_handler(logger):
        path = os.path.join(self.path, log_config['log_path'])
        file_handler = logging.FileHandler(path)

        self.set_logger_level(file_handler)
        self.set_logger_formatter(file_handler)

        logger.addHandler(file_handler)

    def get_logger_format(self):
        return LOG_FORMAT

    def set_logger_formatter(self, logger):
        log_format = self.get_logger_format()
        formatter = logging.Formatter(log_format)
        logger.setFormatter(formatter)

    def set_logger_level(self, logger):
        log_config = self.config
        verbosity = int(log_config['verbosity'])
        try:
            level = VERBOSITY_LEVELS[verbosity]
        except KeyError:
            msg = 'Verbosity level {l} is not in [1, 2, 3, 4]'
            raise ConfigError(msg.format(verbosity))
        logger.setLevel(level)

    def check_config(self):
        log_to_file = bool(self.config['log_to_file'])
        has_path = self.path is not None

        if log_to_file and not has_path:
            msg = (
                'Log to file flag was set to true but no path for log file'
                ' was given.')
            raise ConfigError(msg)

