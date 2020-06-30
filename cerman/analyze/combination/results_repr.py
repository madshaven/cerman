#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Define representation of simulation results:
    info, get-function, format, abbreviation, label, unit, scale, and symbol.

    Define `ResultRepr` telling how to format e.g. value strings and labels.

    All `ResultRepr` are added to a dict `resultr`.

    Define `get_result_key_info` giving info on all parameters.
'''

# General imports
import logging
from collections import OrderedDict

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# idea: add getter function to config
#       (the function may be defined somewhere else)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       CONFIGURATIONS                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# define how simulation results should be represented

configurations = OrderedDict()
configurations['speed'] = {
    'info': 'The average propagation speed, from the whole simulation.',
    'abbr': 'psa',
    'label': 'Average speed',
    'symbol': r'$v$',
    'fmt'  : '2.1f',
    'unit' : 'mm/\u00B5s',
    'scale': 1e-3,
    'title': r'Average propagation speed',
    }
configurations['speed_Q1'] = {
    'info': 'The average propagation speed, from the first quarter of the gap.',
    'abbr': 'pss',
    'label': 'First quarter speed',
    'symbol': r'$v_{q1}$',
    'fmt'  : '2.1f',
    'unit' : 'mm/\u00B5s',
    'scale': 1e-3,
    'title': r'Propagation speed, first quarter',
    }
configurations['speed_Q2Q3'] = {
    'info': 'The average propagation speed, from the middle half of the gap.',
    'abbr': 'psm',
    'label': 'Middle half speed',
    'symbol': r'$v_{q23}$',
    'fmt'  : '2.1f',
    'unit' : 'mm/\u00B5s',
    'scale': 1e-3,
    'title': r'Propagation speed, middle half',
    }
configurations['speed_Q4'] = {
    'info': 'The average propagation speed, from the last quarter of the gap.',
    'abbr': 'pse',
    'label': 'Last quart speed',
    'symbol': r'$v_{q4}$',
    'fmt'  : '2.1f',
    'unit' : 'mm/\u00B5s',
    'scale': 1e-3,
    'title': r'Propagation speed, last quarter',
    }
configurations['first_avalanche_time'] = {
    'info': 'The time until the first critical avalanche occurs.',
    'abbr': 'ita',
    'label': 'Time to first avalanche',
    'symbol': r'$t$',
    'fmt'  : '2.1f',
    'unit' : r'ns',
    'scale': 1e9,
    'title': r'Inception time, first avalanche',
    }
configurations['first_added_avalanche_time'] = {
    'info': 'The time until the first critical avalanche is added as a streamer head.',
    'abbr': 'itaa',
    'label': 'Time to first added avalanche',
    'symbol': r'$t$',
    'fmt'  : '2.1f',
    'unit' : r'ns',
    'scale': 1e9,
    'title': r'Inception time, added avalanche',
    }
configurations['first_z_change_time'] = {
    'info': 'The time until the added streamer head changes z-position.',
    'abbr': 'itz',
    'label': 'Time to first propagation',
    'symbol': r'$t$',
    'fmt'  : '2.1f',
    'unit' : r'ns',
    'scale': 1e9,
    'title': r'Inception Time, first Z change',
    }
configurations['sim_time'] = {
    'info': 'The total time of the simulated experiment.',
    'abbr': 'st',
    'label': 'Simulated time',
    'symbol': r'$t$',
    'fmt'  : '4.1f',
    'unit' : '\u00B5s',
    'scale': 1e6,
    'title': r'Total simulation time',
    }
configurations['cpu_time'] = {
    'info': 'The total time used by the CPU for this simulation.',
    'abbr': 'ct',
    'label': 'CPU time',
    'symbol': r'$t$',
    'fmt'  : '4.1f',
    'unit' : r'hrs',
    'scale': 1 / (60 * 60),
    'title': r'Total CPU time',
    }
configurations['dz_std'] = {
    'info': 'The standard deviation of the discrete changes in z-direction.',
    'abbr': 'jds',
    'label': 'Jump distance deviation',
    'symbol': r'$-$',
    'fmt'  : '2.1f',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'title': r'Jump distance, std dev',
    }
configurations['dz_max'] = {
    'info': 'The maximum of the discrete changes in z-direction.',
    'abbr': 'jdm',
    'label': 'Max jump distance',
    'symbol': r'$-$',
    'fmt'  : '2.1f',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'title': r'Jump distance, max',
    }
configurations['dz_average'] = {
    'info': 'The average of the discrete changes in z-direction.',
    'abbr': 'jda',
    'label': 'Average jump distance',
    'symbol': r'$-$',
    'fmt'  : '2.1f',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'title': r'Average jump distance',
    }
configurations['first_z_change_val'] = {
    'info': 'The magnitude of the first discrete change in z-direction.',
    'abbr': 'jdf',
    'label': 'First jump distance',
    'symbol': r'$-$',
    'fmt'  : '2.1f',
    'unit' : '\u00B5m',
    'scale': 1e6,
    'title': r'First jump distance',
    }
configurations['gap_fraction'] = {
    'info': 'The fraction of the gap propagated before stopping.',
    'abbr': 'gf',
    'label': 'Gap fraction',
    'symbol': r'$-$',
    'fmt'  : '2.1f',
    'unit' : r'%',
    'scale': 1e2,
    'title': r'Gap fraction propagated',
    }
configurations['dz_sum'] = {
    'info': 'The distance of the gap propagated before stopping.',
    'abbr': 'ls',
    'label': 'Propagation length',
    'symbol': r'$-$',
    'fmt'  : '2.1f',
    'unit' : r'mm',
    'scale': 1e3,
    'title': r'Propagation length',
    }


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       UPDATE AND MAP                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# map abbreviations and keys to keys
map_abbr = {v['abbr']: k for k, v in configurations.items()}
map_keys = {k: k for k, v in configurations.items()}
map_keys = dict(map_abbr, **map_keys)
if len(map_abbr) != len(configurations):
    raise SystemExit('ERROR! Implementation. abbr vs config')


class ResultRepr(object):

    def __init__(self, key, abbr, label, info, fmt, unit='',
                 scale=1, symbol=None):

        self.key   = key
        self.abbr  = abbr
        self.label  = label
        self.info  = info
        self.fmt   = fmt
        self.unit  = unit
        self.scale = scale
        self.symbol = symbol

    def _get_val(self, data):
        ''' Return raw value. '''
        return data[self.key]

    def get_val(self, data=None, val=None, scale=None):
        ''' Return scaled value. '''
        scale = self.scale if (scale is None) else scale
        val = self._get_val(data) if (data is not None) else val
        val = (val * scale) if (val is not None) else val
        return val

    def get_val_str(self, data=None, val=None, fmt=None, scale=None,
                    unit=None, brace=False, symbol=True):
        ''' Return scaled value as formatted string. E.g. for legend.

        Parameters
        ----------

        data :      dict
                    simulation result data
        val :       float or bool
                    defaults to the value in `data`
        fmt :       str
                    optional, specify how to format the string
        scale :     float
                    optional, specify the scaling of the value
        unit :      str
                    optional, specify the unit of the value
        brace :     bool
                    optional, add braces around the unit
        symbol :    str or bool
                    optional, add the (default) symbol of the value

        returns
        -------
        out :       str
                    scaled value, possibly with unit and symbol
        '''

        # get value
        if (data is not None):
            val = self._get_val(data)

        # fix special cases
        val = 'Enabled' if (val is True) else val
        val = 'Disabled' if (val is False) else val
        if val is None:
            return str(val)

        # chose input or defaults
        scale = self.scale if scale is None else scale
        unit = self.unit if unit is None else unit
        fmt = self.fmt if fmt is None else fmt

        # format output
        out = '{:{}}'.format(val * scale, fmt)
        if brace and unit:
            unit = '[' + unit + ']'
        if unit:
            out = out + ' ' + unit
        if symbol:
            symbol = self.symbol if symbol is True else symbol
            out = symbol + ' ' + out
        return out

    def get_label(self, label=None, unit=None, brace=True):
        ''' Return label, e.g. for axis. '''
        unit = self.unit if unit is None else unit
        out = self.label if label is None else label
        if brace and unit:
            unit = '[' + unit + ']'
        if unit:
            out = out + ' ' + unit
        return out


# Note: This dict is one of the main function of this file
resultr = OrderedDict()
for key, config in configurations.items():
    resultr[key] = ResultRepr(
        key   = key,
        abbr  = config['abbr'],
        info  = config['info'],
        label  = config['label'],
        fmt   = config['fmt'],
        unit  = config['unit'],
        scale = config['scale'],
        symbol = config['symbol'],
        )


def get_result_key_info(keys=None):
    # return a string containing key, abbreviation and info
    # all available keys are used as default

    logger.debug('Got keys:' + str(keys))
    # use keys if given, else use all keys in dict, in correct order
    if keys is None:
        keys = list(resultr)

    # map_keys contains both keys and abbreviations
    invalid_keys = [k for k in keys if (k not in map_keys)]
    keys = [k for k in keys if (k in map_keys)]
    keys = [map_keys[k] for k in keys]  # change abbreviations to keys

    logger.debug('Using keys:' + str(keys))
    if invalid_keys:
        msg = 'The following keys are invalid:' + str(invalid_keys)
        logger.warning(msg)

    # get the longest needed string length for key and abbreviation
    l_max = max(len(key) + len(resultr[key].abbr) for key in keys)

    msg = '{0:<{1}} ({2:}) - {3:}\n'
    out = ''
    for key in keys:
        abbr = resultr[key].abbr
        info = resultr[key].info
        lk = l_max - len(abbr)
        out += msg.format(key, lk, abbr, info)

    out = out[:-1]  # remove final new line
    return out


#
