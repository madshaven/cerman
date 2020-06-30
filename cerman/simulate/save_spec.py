#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This module contains classes for saving data.

    SaveSpec:

    Main simulation loads all data from "save dict".
    The dict is updated with custom dict given by user input.
    All "enabled" speccs are added.
    User input speccs are enabled by default.
    Default speccs are disabled by default.

    save_spec_dicts: Defines a save spec, used for initiation of the class.
    SaveSpecs: Stores and manages each `SaveSpec`
'''

# General imports
import logging
import pickle
from pathlib import Path
import datetime
from collections import OrderedDict

# Import from project files
from .sim_data import save_sets as sim_data_save_sets

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# idea: Add save.finalize()
#       - clean up any stuff that is needed.
#       - push any unsaved data to file.
#       - write terminator/end-of-file text to file
#       - disable future saving

# this is how dumped files are organized
DUMP_FILE_INFO = \
'''This file is organized as follows:
- header (dict) with keys:
    - time_now (current time)
    - file_type (method for saving)
    - file_info (this string)
    - save_spec_to_dict (the parameters for this save spec)
    - sim_input (the parameters for this simulation)
- single data dump
- single data dump
- ...
- single data dump
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       SAVE SPEC DICTS                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
''' These are the default save spec dicts.
    A `save_spec_dict` is used to initiate a `SaveSpec`.
    Each defines a selection data to extract, and when to extract it.
    Note, the `save_set_key` points to a `save_set`.
'''
save_spec_dicts = OrderedDict()

# The `save_set` named `stat` specifies a small number parameters to save.
# This is sufficient to plot e.g. shadow and speed
# Note, other methods tend to not capture all added heads
save_spec_dicts['stat'] = {
    'style': 'interval',    # method to use
    'interval': 1,          # how often to save
    'interval_key': 'no',   # how to count how often
    'save_set': [           # what to save
        'no',                       # for reference
        'sim_time',                 # for reference
        'cpu_time',                 # for reference
        'streamer_z_min',           # speed, breakdown, inception
        'streamer_pos_appended',    # shadow, inception
        'seeds_no_critical',        # time to avalanche
        ],
    }

# Save "all" data every 10 raised to n'th "step" (sa#)
# This enables inspection of most simulation details, but requires much space.
save_spec_dicts['sa0'] = {'style': 'interval',
                          'interval_key': 'no',
                          'interval': 1,
                          'n_repeat': 1000,
                          'save_set_key': 'all_reduced'}
# for debugging, it is sometimes nice to save each step, but limit the number
save_spec_dicts['sa0_2'] = dict(save_spec_dicts['sa0'],  # copy the sa0
                                n_repeat=1e2)            # limit times to save
save_spec_dicts['sa1'] = dict(save_spec_dicts['sa0'], interval=1e1)
save_spec_dicts['sa2'] = dict(save_spec_dicts['sa0'], interval=1e2)
save_spec_dicts['sa3'] = dict(save_spec_dicts['sa0'], interval=1e3)
save_spec_dicts['sa4'] = dict(save_spec_dicts['sa0'], interval=1e4)
save_spec_dicts['sa5'] = dict(save_spec_dicts['sa0'], interval=1e5)
save_spec_dicts['sa6'] = dict(save_spec_dicts['sa0'], interval=1e6)

# Save "all" data every 10 raised to negative n'th "seconds" (ta#)
# This enables inspection of most simulation details, but requires much space.
save_spec_dicts['ta13'] = {'style': 'interval',
                           'interval_key': 'sim_time',  # use time as interval
                           'interval': 1e-13,
                           'save_set_key': 'all_reduced'}  # save most data
save_spec_dicts['ta12'] = dict(save_spec_dicts['ta13'], interval=1e-12)
save_spec_dicts['ta11'] = dict(save_spec_dicts['ta13'], interval=1e-11)
save_spec_dicts['ta10'] = dict(save_spec_dicts['ta13'], interval=1e-10)
save_spec_dicts['ta09'] = dict(save_spec_dicts['ta13'], interval=1e-09)
save_spec_dicts['ta08'] = dict(save_spec_dicts['ta13'], interval=1e-08)
save_spec_dicts['ta07'] = dict(save_spec_dicts['ta13'], interval=1e-07)
save_spec_dicts['ta06'] = dict(save_spec_dicts['ta13'], interval=1e-06)
save_spec_dicts['ta05'] = dict(save_spec_dicts['ta13'], interval=1e-05)

# Save "all" data every n'th "percent" of gap propagation (gp#)
# This enables inspection of most simulation details, but requires much space.
save_spec_dicts['gp1'] = {'style': 'interval',
                          'interval_key': 'gap_percent',
                          'interval': 1,
                          'save_set_key': 'all_reduced',}
save_spec_dicts['gp5'] = dict(save_spec_dicts['gp1'], interval=5)
save_spec_dicts['gp10'] = dict(save_spec_dicts['gp1'], interval=10)
save_spec_dicts['gp20'] = dict(save_spec_dicts['gp1'], interval=20)


# Capture the streamer heads, with details, relatively often
# Note, be careful when just capturing upon propagation
save_spec_dicts['streamer'] = {
    'style': 'interval',
    'interval_key': 'gap_percent',
    'interval': 0.1,
    # 'n_repeat': -1,  # repeat infinitely (default)
    'save_set': save_spec_dicts['stat']['save_set'] + [
        'streamer_heads_dict',
        'streamer_heads_dict_appended',
        'streamer_heads_dict_removed',
    ],
}

# capture new streamer positions
save_spec_dicts['shadow'] = {
    'style': 'event',  #
    # 'thres_l': 0,      # by default
    # 'thres_h': None,   # by default
    'save_event': 'streamer_no_appended',  # whenever a head is added
    'save_set': [
        'no',
        'streamer_pos_appended',
        ]
    }
# capture the initiation only
save_spec_dicts['initiation'] = {
    'style': 'event',
    'save_event': 'gap_percent',
    'thres_l': 0,   # start at once
    'thres_h': 2,   # end after 2 percent propagation
    'save_set_key': 'all_reduced'   # save all data
    }
# capture the first occurrences of critical seeds
save_spec_dicts['seeds_critical'] = {
    'style': 'event',
    'n_repeat': 50,
    'save_event': 'seeds_no_critical',
    'save_set_key': 'all_reduced',
    }
# capture a range of iteration numbers
save_spec_dicts['rangeno'] = {
    'style': 'event',
    'save_event': 'no',
    'thres_l': 31,
    'thres_h': 42,
    'save_set_key': 'all_reduced',
    }
# capture final part of propagation
save_spec_dicts['gapfianl'] = {
    'style': 'event',
    'save_event': 'gap_percent',
    'thres_l': 90,
    'save_set_key': 'all_reduced',
    }


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       SAVE SPECS                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class SaveSpecs(object):

    # add the dicts as a class variable
    save_spec_dicts = save_spec_dicts

    # possible methods for choosing how/when to save
    _acceptable_styles = [
        'event',
        'interval',
        ]

    def __init__(self, save_specs_enabled, save_spec_dicts,
                 folder, name, data, mock=False):
        ''' Create object based on content of sim_input. '''

        # initiate logger
        self.logger = logging.getLogger(__name__ + '.SaveSpecs')
        # name of all save specs managed by this instance (save_spec_keys)
        self._keys = set()
        # list of all save specs managed by this instance
        self._save_specs = []
        # set of events trigger saving (always a part of save_set)
        self.save_events = set()
        # defines which keys to save, update each iteration
        self.save_set = set()
        # add custom dicts, if any (user defined)
        self.save_spec_dicts.update(save_spec_dicts)
        # Trim invalid keys (copy needed to modify dict within loop)
        for key, enabled in save_specs_enabled.copy().items():
            if key not in self.save_spec_dicts:
                save_specs_enabled.pop(key)
                msg = f'Removed invalid save spec "{key}".'
                logger.warning('Warning! ' + msg)

        # idea: combine this with the code block above
        # Trim save_spec_dicts, include just the enabled ones
        self.save_spec_dicts = {
            key: self.save_spec_dicts[key]
            for key, enabled in save_specs_enabled.items()
            if enabled or (key in save_spec_dicts)
            }

        # Create all save specs and append them
        for key, save_spec_dict in self.save_spec_dicts.items():
            self.create_and_append(key, save_spec_dict)

        if not mock:
            self.init_dump_file(folder, name)   # create each file for saving
            self.write_header(data)             # write header of each file

        self.logger.debug('Initiated SaveSpecs')
        self.logger.log(5, 'SaveSpecs.__dict__')
        for k, v in self.__dict__.items():
            self.logger.log(5, '  "{}": {}'.format(k, v))

    def create_and_append(self, key, save_spec_dict):
        ''' Initiate the appropriate SaveSpec subclass. '''

        if type(key) is not str:
            msg = f'SaveSpec key "{key}" is not a string. Ignoring.'
            self.logger.warning('Warning! ' + msg)
            return

        style = save_spec_dict.get('style')
        if style not in self._acceptable_styles:
            msg = f'SaveSpec "{key}" has invalid style "{style}". Ignoring.'
            self.logger.warning('Warning! ' + msg)
            return

        if key in self._keys:
            msg = f'SaveSpec key "{key}" is already taken. Skipping.'
            self.logger.warning('Warning! ' + msg)
            return

        self._keys.add(key)

        # Create and append the correct object
        if False:
            pass
        elif style == 'interval':
            self._save_specs.append(SaveSpecInterval(key, save_spec_dict))
        elif style == 'event':
            self._save_specs.append(SaveSpecEvent(key, save_spec_dict))
            self.save_events.add(save_spec_dict['save_event'])
        else:
            msg = f'Warning! Could not find SaveSpec style "{style}"'
            self.logger.warning(msg)

    # The methods below performs the corresponding method
    # for all `_save_specs`. See details in each `SaveSpec`

    def init_dump_file(self, folder, name):
        for save_spec in self._save_specs:
            save_spec.init_dump_file(folder, name)

    def write_header(self, sim_input):
        for save_spec in self._save_specs:
            save_spec.write_header(sim_input)

    def update_save_now(self, sim_data):
        # Check if the criteria for data extraction is fulfilled
        for save_spec in self._save_specs:
            save_spec.update_save_now(sim_data)

    def update_save_set(self):
        # Note, `save_now` should be updated before this point
        self.save_set = self.save_events.copy()
        for save_spec in self._save_specs:
            if save_spec.save_now:
                self.save_set |= save_spec.save_set

    def append_data(self, sim_data):
        for save_spec in self._save_specs:
            if save_spec.save_now:
                save_spec.append_data(sim_data)

    def dump_data(self):
        for save_spec in self._save_specs:
            save_spec.dump_data()


class SaveSpec(object):

    def __init__(self, key, save_spec_dict):
        ''' Define an number of variables to extract from a simulation,
            when to extract them, also manage storing and dumping to file.

            Note
            ----
            Use a subclass to set proper criteria for when to extract data.
            This class by itself just check that `n_repeat` is not exceeded.
        '''
        self.logger = logging.getLogger(__name__ + '.SaveSpec.' + key)

        # name for this save spec
        self.key = key

        # the style defines how to set a criteria for data extraction
        self.style = save_spec_dict.get('style')

        # define max number of saves
        self.n_repeat = save_spec_dict.get('n_repeat', -1)

        # define the variables to be saved / extracted by this instance
        self.save_set_key = save_spec_dict.get('save_set_key', None)
        self.save_set = save_spec_dict.get('save_set', set())
        self.save_set = set(self.save_set)  # ensure set, list may be given
        self.save_set|= sim_data_save_sets[self.save_set_key]
        self.save_set|= set(['no', 'sim_time'])  # always get these

        # Initiate other
        self.data = []          # stores data between dumps to file
        self.save_now = False   # extract data this iteration?
        self.n_saved = 0        # number of saved iterations
        self.fpath = None       # path to save file (see ´init_dump_file`)

        msg = f'Created SaveSpec "{self.key}" with style "{self.style}".'
        self.logger.debug(msg)
        for k, v in self.__dict__.items():
            self.logger.log(5, f'  "{k}": {v}')

    def init_dump_file(self, folder, name):
        ''' Initialize an empty dump file. Create folder if needed.
        '''
        self.folder = folder
        self.name = name
        fpath = '{}/{}_{}.pkl'.format(self.folder, self.name, self.key)
        self.fpath = Path(fpath)

        # Create folder
        if not self.fpath.parent.exists():
            self.fpath.parent.mkdir()

        # Create file
        with self.fpath.open('wb'):
            pass

    def dump(self, data=None, data_lst=[]):
        ''' Dump given data.
        `data` is a single dump,
        `data_lst` is dumped sequentially.
        '''

        if self.fpath is None:
            msg = f'{self.key} : Dump file not initiated, skipping dump.'
            self.logger.warning(msg)
            return

        if data is None:
            data_dump = data_lst
        else:
            data_dump = [data] + data_lst

        with self.fpath.open('ab') as f:
            for data in data_dump:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def write_header(self, sim_input):
        ''' Write a header to the dump file. '''
        header = OrderedDict()
        header['time_now'] = str(datetime.datetime.now())
        header['file_type'] = 'save_raw'
        header['file_info'] = DUMP_FILE_INFO
        header['save_spec_to_dict'] = self.to_dict()
        header['sim_input'] = sim_input

        self.dump(header)

    # note, same signature as subs
    def update_save_now(self, sim_data=None):
        # Check if the criteria for data extraction is fulfilled
        self.save_now = (self.n_repeat == -1) or (self.n_saved < self.n_repeat)

    def append_data(self, sim_data):
        ''' Extract relevant data from 'sim_data' to a dict
        and append it to 'data'.
        '''
        self.n_saved += 1
        tmp_data = {}
        for key in self.save_set:
            tmp_data[key] = sim_data[key]
        self.data.append(tmp_data)

    def dump_data(self):
        ''' Dump everything in 'data' to a file, then clear 'data'.
        '''
        if self.data:
            self.dump(data_lst=self.data)
            self.data = []

    def to_dict(self):
        # return a dict of the arguments
        self_dict = {}
        self_dict['key']          = self.key
        self_dict['style']        = self.style
        self_dict['n_repeat']     = self.n_repeat
        self_dict['save_set_key'] = self.save_set_key
        self_dict['save_set']     = self.save_set
        return self_dict


class SaveSpecInterval(SaveSpec):
    def __init__(self, key, save_spec_dict):
        ''' This class is used for saving given intervals
            of a monotonically increasing variable.
        '''
        # Use constructor of superclass to set most variables
        super().__init__(key, save_spec_dict)

        # use a variable that is monotonically increasing
        self.interval_key = save_spec_dict['interval_key']

        # define how often to save
        self.interval = save_spec_dict['interval']

        # define next time to save (updated when a save is performed)
        self.next_save = 0  # Add the very first incidence

    def update_save_now(self, sim_data):
        # Check if the criteria for data extraction is fulfilled
        super().update_save_now(sim_data)
        if (self.save_now):  # a 'no' from super means 'no'
            # Do this unsafe on purpose, correct keys expected here
            _now = sim_data[self.interval_key]
            self.save_now = (self.next_save <= _now)
        self.logger.log(5, f'{self.key} : Save_now is {self.save_now}')

    def append_data(self, sim_data):
        # Append new data, update when next save is
        super().append_data(sim_data)
        # keep fixed intervals, do not use `now`+`interval`
        self.next_save += self.interval

    def to_dict(self):
        # return a dict of the arguments
        self_dict = super().to_dict()
        self_dict['interval_key'] = self.interval_key
        self_dict['interval'] = self.interval
        return self_dict


class SaveSpecEvent(SaveSpec):
    ''' This class is used for saving when
        a given variable is within given thresholds.

    Essentially: save_now = (thres_l < sim_data[save_event] <= thres_h)

    Notes
    -----
    `save_event` is a key of an int variable that can be extracted.
    `thres_l` sets the lower threshold, defaults to 0.
    `thres_h` sets the higher threshold, defaults to None (inf).
    If a threshold is `None`, it is ignored.

    Examples
    --------
    streamer_no_appended > 0    -->  Streamer growth
    seeds_no_critical > 0       -->  Avalanche
    0 < gap_percent < 5         -->  First 5 % of propagation only
    '''

    # idea: add saveEvents, chaining multiple events, and/or

    def __init__(self, key, save_spec_dict):
        # Use constructor of superclass to set variables
        super().__init__(key, save_spec_dict)

        # idea: use pop to want on extra values
        #       make a copy to be sure

        # Style specific defaults
        self.save_event = save_spec_dict['save_event']  # error if missing
        self.thres_l = save_spec_dict.get('thres_l', 0)
        self.thres_h = save_spec_dict.get('thres_h', None)
        if (self.save_event == ''):
            self.logger.error('Save_event not defined')

        if (self.save_event not in sim_data_save_sets['all']):
            msg = f'Save_event "{self.save_event}" not defined'
            self.logger.error('Error! ' + msg)

    def update_save_now(self, sim_data):
        # Check if the criteria for data extraction is fulfilled
        super().update_save_now(sim_data)
        if (self.save_now):  # a 'no' from super means 'no'
            # get value, provoke error on mistake
            val = sim_data[self.save_event]
            # check borders, ignore any border set to `None`
            # note: latter part of OR is not evaluated if the first is True
            is_above = (self.thres_l is None) or (val > self.thres_l)
            is_below = (self.thres_h is None) or (val <= self.thres_h)
            # flag saving if both borders are respected
            self.save_now = is_above and is_below

    def to_dict(self):
        # return a dict of the arguments
        self_dict = super().to_dict()
        self_dict['save_event'] = self.save_event
        self_dict['thres_l'] = self.thres_l
        self_dict['thres_h'] = self.thres_h
        return self_dict

#
