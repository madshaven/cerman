#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This module contains functionality for extracting data from the simulation.

The class `SimData` is used to extract data from the simulation variables.
Classes used to store data get their data through the `SimData` class.

The fist part specifies a number of variables that can be saved.
The second part is a class `SimData`, which performs several tasks.
`SimData` is the class that
keeps track of simulation iteration number and time spent.
It also writes information and the status of the simulation,
as well as decides when a simulation is to be stopped.
Furthermore, the class acts as an interface
between the simulation variables and the save specification.

Data that may be saved during simulation is defined by a specific `save_key`.
A `save_set` is a set of such `save_key`s.
And `save_sets` is a dictionary of `save_set`s, indexed by a `save_set_key`.
The set `save_set_all` defines all possible `save_key`s.
Commonly used `save_set`s are defined in `save_sets`.
'''

# General imports
import logging
import time

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# idea: save python lists instead of lists of numpy arrays
#       (saves space, especially for sparse lists.)
# idea: consider Pandas, or another save format
# idea: use iteration number is used for indexing of saved data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       SAVE KEYS AND SAVE SETS                                       #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A `save_key` refers to something that can be saved from SimData.
# A `save_set` is a set of such keys.
# save_set = {save_key, }
# `save_sets` is a dict defining a number of save sets.
# save_sets[save_set_key] = save_set
# The save set 'all' contains all keys that can be used for extraction.

# idea: is it possible to define keys with their getters?

# Define basic save sets, group for readability, and later reference
save_sets = {}

# define the empty set for convenience
save_sets[None] = set()

save_sets['simulation_loop'] = {  # managed by SimData.`_update_loop`
    'no',
    'sim_dt',
    'sim_time',
    'cpu_dt',
    'cpu_time',
    }
save_sets['simulation_derived'] = {
    'info',             # managed by SimData.`_update_other`
    'speed_avg',        # managed by SimData.`_update_other`
    'gap_percent',      # managed by SimData.`_update_other`
    'time_since_avalanche',  # managed by SimData.`_update_other`
    'status',           # managed by SimData.`update_continue`
    'continue',         # managed by SimData.`update_continue`
    }
save_sets['roi'] = {
    'roi',  # managed by SimData.`extract_data` through `dict_getters`
    }
save_sets['streamer_other'] = {  # managed by SimData.`_copy_ints`
    'streamer_z_min',
    'streamer_r_max',
    }
save_sets['streamer_no'] = {  # managed by SimData.`_copy_ints`
    'streamer_no',
    'streamer_no_appended',
    'streamer_no_removed',
    'streamer_no_appended_all',
    'streamer_no_removed_all',
    }
save_sets['streamer_cum'] = set(  # managed by SimData.`_update_other`
    'streamer_cum' + sk[len('streamer_no') :]
    for sk in save_sets['streamer_no']
    )
save_sets['streamer_pos'] = {  # managed by SimData.`extract_data`
    'streamer_pos',
    'streamer_pos_appended',
    'streamer_pos_removed',
    'streamer_pos_appended_all',
    'streamer_pos_removed_all',
    }
save_sets['streamer_heads_dict'] = {  # managed by SimData.`extract_data`
    'streamer_heads_dict',
    'streamer_heads_dict_appended',
    'streamer_heads_dict_removed',
    'streamer_heads_dict_appended_all',
    'streamer_heads_dict_removed_all',
    }
save_sets['seeds_other'] = {  # managed by SimData.`_copy_ints`
    'seeds_Q_max',
    'seeds_dQ_max',
    'seeds_ds_max',
    'seeds_charge',
    'seeds_charge_gen',
    'seeds_charge_rem',
    }
save_sets['seeds_no'] = {  # managed by SimData.`_copy_ints`
    'seeds_no',
    'seeds_no_to_append',
    'seeds_no_ion',
    'seeds_no_electron',
    'seeds_no_avalanche',
    'seeds_no_critical',
    'seeds_no_behind_roi',
    'seeds_no_in_streamer',
    'seeds_no_to_remove',
    }
save_sets['seeds_cum'] = set(  # managed by SimData.`_update_other`
    'seeds_cum' + sk[len('seeds_no') :]
    for sk in save_sets['seeds_no']
    )
save_sets['seeds_pos'] = {   # managed by SimData.`extract_data`
    'seeds_pos',
    'seeds_pos_ion',
    'seeds_pos_electron',
    'seeds_pos_avalanche',
    'seeds_pos_critical',
    'seeds_pos_behind_roi',
    'seeds_pos_in_streamer',
    'seeds_pos_to_append',
    'seeds_pos_to_remove',
    }
save_sets['seeds_Q'] = {  # managed by SimData.`extract_data`
    'seeds_Q',
    'seeds_Q_ion',
    'seeds_Q_electron',
    'seeds_Q_avalanche',
    'seeds_Q_critical',
    'seeds_Q_behind_roi',
    'seeds_Q_in_streamer',
    'seeds_Q_to_remove',
    }
save_sets['seeds_dQ'] = {  # managed by SimData.`extract_data`
    'seeds_dQ',
    'seeds_dQ_ion',
    'seeds_dQ_electron',
    'seeds_dQ_avalanche',
    'seeds_dQ_critical',
    'seeds_dQ_behind_roi',
    'seeds_dQ_in_streamer',
    'seeds_dQ_to_remove',
    }
save_sets['seeds_bools'] = {  # managed by SimData.`extract_data`
    'seeds_is_ion',
    'seeds_is_electron',
    'seeds_is_critical',
    'seeds_is_avalanche',
    'seeds_is_behind_roi',
    'seeds_is_in_streamer',
    }

# Define a set of all possible save keys
save_set_all = set(sk for ssk, ss in save_sets.items() for sk in ss)
save_sets['all'] = save_set_all.copy()

# do not keep seed data that can be reconstructed from `seeds_bools`
save_sets['all_reduced'] = set(
    sk for sk in save_sets['all']  # iterate all keys
    if sk not in set(              # create a set to exclude
        'seeds_' + k1 + k2[len('seeds_is') :]  # build keys to exclude
        for k1 in ['pos', 'Q', 'dQ']           # exclude categories
        for k2 in save_sets['seeds_bools']     # exclude details
        )
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DATA EXTRACTION                                               #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class SimData(object):
    def __init__(self):


        # Initialize dict for simulation data.
        self.data = {}
        for key in save_set_all:
            # Set all values to 0 initially.
            # This enables the use of += later
            # collections.defaultdict could be used instead,
            # but not will give an error,
            # which is preferable, in some cases
            self.data[key] = 0

        self.process_start = time.process_time()
        logger.debug('Initiated SimData')
        logger.log(5, 'SimData.__dict__')
        for k, v in self.__dict__.items():
            logger.log(5, '  "{}": {}'.format(k, v))

    def set_stop_conditions(self, sim_input):
        ''' Use simulation input to set the stop conditions. '''

        # idea: implement None as no action
        #       missing keys to be set to None

        keys = [  # keys used by 'update_continue'
            'stop_iteration',
            'stop_z_min',
            'stop_heads_max',
            'stop_seeds_no_electron',
            'stop_sim_time',
            'stop_cpu_time',
            'stop_cpu_dt',
            'stop_time_since_avalanche',
            'stop_speed_avg',
            ]

        # validate input
        input_keys = [k for k in sim_input if 'stop_' in k]
        missing_input = [k for k in keys if k not in input_keys]
        unused_input = [k for k in input_keys if k not in keys]

        if missing_input:
            logger.error('Error. Stop condition(s) missing:')
            logger.error(', '.join(missing_input))
            return

        if unused_input:
            logger.warning('Warning. Unable to use stop condition(s):')
            logger.warning(', '.join(unused_input))

        self.stop_conditions = {k: sim_input[k] for k in keys}

    def update_continue(self):
        ''' Check whether the loop should `continue`. Write `status`.
        '''
        # idea: make this a specific class
        # idea: consider making stop conditions a tuple of:
        #       (data_key, sc_key, operator, string)
        #       (get inspiration from SaveSpecEvent?)
        # idea: consider adding a 'minimum' requirement
        #       (e.g. minimum iterations/time/propagation before stopping)

        d = self.data
        s = self.stop_conditions
        msg = ''
        d['continue'] = False

        if False:   # just to get symmetry
            pass

        elif d['no'] >= s['stop_iteration']:
            msg = 'Max iteration number reached. ({:d})'
            msg = msg.format(d['no'])

        elif d['streamer_z_min'] <= s['stop_z_min']:
            msg = 'Streamer reached bottom electrode. ({:#0.4g})'
            msg = msg.format(d['streamer_z_min'])

        elif d['streamer_no'] >= s['stop_heads_max']:
            msg = 'Too many active streamer heads. ({:d})'
            msg = msg.format(d['streamer_no'])

        elif d['seeds_no_electron'] <= s['stop_seeds_no_electron']:
            msg = 'Too few electron seeds. ({:d})'
            msg = msg.format(d['seeds_no_electron'])

        elif d['sim_time'] >= s['stop_sim_time']:
            msg = 'Sim time exceeded. ({:#0.4g})'
            msg = msg.format(d['sim_time'])

        elif d['cpu_time'] >= s['stop_cpu_time']:
            msg = 'CPU time exceeded. ({:#0.4g})'
            msg = msg.format(d['cpu_time'])

        elif d['cpu_dt'] >= s['stop_cpu_dt']:
            msg = 'CPU dt exceeded. ({:#0.4g})'
            msg = msg.format(d['cpu_dt'])

        elif d['time_since_avalanche'] >= s['stop_time_since_avalanche']:
            msg = 'Avalanche time exceeded. ({:#0.4g})'
            msg = msg.format(d['time_since_avalanche'])

        # Note the added requirement. This is inspiration to add a minimum.
        elif ((d['gap_percent'] >= 1) and  # added to ensure start
                (d['speed_avg'] <= s['stop_speed_avg'])):
            msg = 'Low average speed. ({:#0.4g})'
            msg = msg.format(d['speed_avg'])

        else:
            d['continue'] = True
            msg = 'Simulation running.'

        d['status'] = msg

    def update_data(self, sim_vars):
        ''' Update/extract data from the simulation.

        Used for information that is fast/cheap to copy.

        Parameters
        ----------
        sim_vars     :  dict
                        streamer, seeds, roi, needle
        '''

        streamer = sim_vars['streamer']
        seeds    = sim_vars['seeds']
        needle   = sim_vars['needle']

        # advance the iteration loop number and save time spent
        self._update_loop(time_spent=seeds.dt)

        # copy int variables, easier and probably faster than checking
        self._copy_ints(streamer, seeds)

        # update cumulative and other derivative/calculated data
        self._update_other(needle.d)

    def extract_data(self, sim_vars, save_set=save_set_all):
        ''' Update/extract data from the simulation.

        Used for information that is slow/expensive to copy.

        Parameters
        ----------
        sim_vars     :  dict
                        dictionary containing all simulation variables
        save_set     :  set of str
                        Meta data details and these 'save_set' is extracted
        '''

        # extract from streamer
        streamer = sim_vars['streamer']

        # extract streamer positions
        keys = [k for k in save_set if k in save_sets['streamer_pos']]
        for k in keys:
            self.data[k] = getattr(streamer, k[len('streamer_'):])

        # extract streamer dictionaries
        keys = [k for k in save_set if k in save_sets['streamer_heads_dict']]
        for k in keys:  # get dict lists instead of heads class lists
            k_ = 'heads_dict' + k[len('streamer_heads_dict'):]
            self.data[k] = getattr(streamer, k_)

        # extract from seeds
        seeds = sim_vars['seeds']
        ss = set().union(
            save_sets['seeds_pos'],
            save_sets['seeds_Q'],
            save_sets['seeds_dQ'],
            save_sets['seeds_bools'],
            )
        keys = [k for k in save_set if k in ss]
        for k in keys:  # copy these since they may be views
            self.data[k] = getattr(seeds, k[len('seeds_'):]).copy()

        # copy arrays, see set_array_getters
        keys = [k for k in save_set if k in self.array_getters]
        for k in keys:
            self.data[k] = self.array_getters[k]()

        # copy dicts, see set_dict_getters
        keys = [k for k in save_set if k in self.dict_getters]
        for k in keys:
            self.data[k] = self.dict_getters[k]()

    def _update_loop(self, time_spent):
        ''' Update variables related to the simulation loop. '''
        self.data['no']       += 1
        self.data['sim_dt']   = time_spent
        self.data['sim_time'] += time_spent
        self.data['cpu_dt']   = time.process_time() - self.data['cpu_time']
        self.data['cpu_time'] = time.process_time() - self.process_start

    def _copy_ints(self, streamer, seeds):
        # It should be faster to just copy all ints,
        # than to check whether to copy.
        d = self.data

        # streamer
        for sk in save_sets['streamer_other']:
            d[sk] = getattr(streamer, sk[len('streamer_') :])
        for sk in save_sets['streamer_no']:
            d[sk] = getattr(streamer, sk[len('streamer_') :])

        # seeds
        for sk in save_sets['seeds_other']:
            d[sk] = getattr(seeds, sk[len('seeds_') :])
        for sk in save_sets['seeds_no']:
            d[sk] = getattr(seeds, sk[len('seeds_') :])

    def _update_other(self, needle_d):
        # update dependent data

        d = self.data

        # streamer
        for sk in save_sets['streamer_cum']:
            sk_ = 'streamer_no' + sk[len('streamer_cum') :]
            d[sk] += d[sk_]

        # seeds
        for sk in save_sets['seeds_cum']:
            sk_ = 'seeds_no' + sk[len('seeds_cum') :]
            d[sk] += d[sk_]

        d['gap_percent'] = (1 - d['streamer_z_min'] / needle_d) * 100
        d['speed_avg'] = (needle_d - d['streamer_z_min']) / d['sim_time']

        # add info string
        # idea: make info string configurable
        # note: formatting numpy arrays as 'floats' cause problems
        info = '{:2}; {:4.2f}; {:2.2f}%; {:4.1f}; {:4.2f}; {:4.2f}'

        info = info.format(
            d['streamer_no'],
            d['speed_avg'] * 1e-3,
            d['gap_percent'],
            d['seeds_Q_max'],
            d['seeds_dQ_max'],
            d['seeds_ds_max'] * 1e6,
            )
        d['info'] = info

        d['time_since_avalanche'] += d['sim_dt']
        if d['seeds_no_critical'] > 0:
            d['time_since_avalanche'] = 0

    def set_array_getters(self, streamer, seeds):
        ''' Define functions for getting arrays. '''

        g = {}  # getters
        self.array_getters = g

    def set_dict_getters(self, roi, streamer, seeds):
        ''' Define functions for getting dictionaries. '''

        g = {}  # getters
        g['roi'] = roi.to_dict
        self.dict_getters = g


#
