#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Configure logging functionality for simulation script.
'''

# General imports
from pathlib import Path
import datetime
import logging                  # important to import this last

# Import from project files
from ..tools import json_dumps

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# create formatters
# %(filename)s, %(levelname)s, %(funcName)s():, %(lineno)d, %(message)s
fmt_nlm  = logging.Formatter('%(levelname)s %(name)s: %(message)s')
fmt_lm   = logging.Formatter('%(levelname)s - %(message)s')
fmt_tnlm = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fmt_lnm  = logging.Formatter('%(levelname)s - %(filename)s: %(message)s')


class LogFormatter(logging.Formatter):
    ''' Add iteration number to log records. '''

    def __init__(self, logger, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logger
        self.name = name

    def _no(self):
        # Note: `iteration_no` must be set manually elsewhere
        return self.logger.iteration_no

    def format(self, record):
        # add name and iteration number to the record
        out = logging.Formatter.format(self, record)
        return f'{self.name:s}:{self._no():d}: {out:s}'


def init_logger(logger, log_name, log_folder, log_level_file,
                log_level_console, data):
    ''' Initiate logger.
    Create log file with header, add file handler, and add console handler.

    The predefined logging levels are:
        - 50 - CRITICAL
        - 40 - ERROR
        - 30 - WARNING
        - 20 - INFO
        - 10 - DEBUG
        - 00 - NOTSET

    Parameters
    ----------
    logger            : logger instance
                        instance to initiate
    log_name          : str
                        name of log file
    log_folder        : str
                        log file folder
    log_level_file    : int
                        level of logging to file
    log_level_console : int
                        level of logging to console
    data              : dict
                        data to be dumped at start of log
    '''

    # Close and remove any existing handlers
    # print('All handlers:', logging.Logger.manager.loggerDict.keys())
    # print('Handlers:', logger.handlers)
    # for h in logging.getLogger().handlers:
    for h in logger.handlers:
        logger.warning('Removed existing log handler!')
        h.close()
        logger.removeHandler(h)

    # # Set minimum logging level
    logger.setLevel(min(log_level_file, log_level_console))

    # Set path and create folder, if required
    folder = Path(log_folder)
    fpath = Path(f'./{folder}/{log_name}.log')
    if not folder.is_dir():
        folder.mkdir()

    # Create log file and write header
    with fpath.open('w') as f:
        f.write(str(fpath.resolve()) + '\n\n')
        f.write('Log file for streamer simulation.\n')
        f.write('Created {}\n\n'.format(datetime.datetime.now()))
        f.write('Data input to simulation:\n')
        f.write(json_dumps(data))
        f.write('\n')
        f.write('\n')

    # Initiate log formatter that also logs iteration number
    logger.iteration_no = 0
    fmt_info = LogFormatter(logger, log_name, '%(message)s')
    fmt_debug = LogFormatter(
        logger, log_name, '%(filename)s:%(lineno)d: %(message)s')

    # Add file handler
    fh = logging.FileHandler(str(fpath), mode='a')
    fh.setFormatter(fmt_info)
    if log_level_file < 20:
        fh.setFormatter(fmt_debug)
    fh.setLevel(log_level_file)
    logger.addHandler(fh)
    logger.debug(f'File handler added at level {log_level_file}')

    # Add console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt_info)
    if log_level_console < 20:
        ch.setFormatter(fmt_debug)
    ch.setLevel(log_level_console)
    logger.addHandler(ch)
    logger.debug(f'Console handler added at level {log_level_console}')

    # return the logger, file handler, and console handler
    return logger, fh, ch

#
