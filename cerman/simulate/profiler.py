#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This module contains a simple class for profiling.
'''

# General imports
import logging
import cProfile
from pathlib import Path

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class Profiler(object):
    def __init__(self, name, folder, enabled=False):

        # set correct name, folder, and path
        self.folder = folder
        self.name = name
        fpath = f'{self.folder}/{self.name}.profile'
        self.fpath = Path(fpath)

        # create folder if needed
        if not self.fpath.parent.exists():
            self.fpath.parent.mkdir()

        # create and set up the profiler
        self.profiler = cProfile.Profile()
        self.enabled = enabled
        if enabled:
            self.profiler.enable()
            logger.debug('Profiler enabled')
            logger.log(5, 'Profiler file {}'.format(str(self.fpath)))
        else:
            logger.debug('Profiler disabled')

        logger.debug('Initiated Profiler')
        logger.log(5, 'Profiler.__dict__')
        for k, v in self.__dict__.items():
            logger.log(5, '  "{}": {}'.format(k, v))

    def dump_stats(self):
        if self.enabled:
            self.profiler.dump_stats(str(self.fpath))
            self.profiler.enable()  # dumping disables the profiler

    def enable(self):
        self.enabled = True
        self.profiler.enable()

    def disable(self):
        self.enabled = False
        self.profiler.disable()

#
