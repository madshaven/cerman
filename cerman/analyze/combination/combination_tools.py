#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Functions used by several types of plots.
'''

# General imports
import numpy as np
from pathlib import Path
import logging
from scipy import interpolate
from collections import OrderedDict

# Import from project files
from . import results_repr
from .. import sim_parameters

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

x_cfg = sim_parameters.configurations
x_map = sim_parameters.map_keys
x_dep = sim_parameters.dependencies
y_cfg = results_repr.configurations
y_map = results_repr.map_keys


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       K_DICT TOOLS                                                  #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
'''
    A k-dict defines how a simulation parameter (k_key) and its value (k_val)
    should be included in a plot (mode).
    k_dict: {k_key: {k_val: mode}}

    Each combination of (k_key, k_val) corresponds to a set of files.
    The mode define how to treat the results from these files:
    'color' :   Use colors to differentiate.
    'marker' :  Use markers.
    'include' : Include this combination.
    'exclude' : Exclude this combination.
'''

def k_dict_gen(k_keys, k_dict=None, modes=None, inptdd=None):
    ''' Generate k_dict to use for plotting, recursively.

        Initiation only:
        Generate a k_dict where everything are set to 'exclude'.

        Create a k_dict for each k_key.
        Each value of that k_key is set to the first mode.
        Call this function using the rest of the k_keys and modes.

        Each recursive call trims one k_key and one mode.
        A copy of the k_dict is yielded when there are no k_keys.

        Parameters
        ----------
        k_keys :    list(str)
                    name of simulation input parameters
        k_dict :    dict
                    part of the recursive algorithm
        modes :     list(str)
                    sequence of plot modes
        inptdd :    dictionary
                    analyze to find k_vals of each k_key

        Yields
        ------
        k_dict :    {k_key: {k_val: string}}
    '''
    # todo: consider how to act when k_keys are longer than modes.

    if modes is None:               # default modes
        modes = ['color', 'marker', 'include']
    mode = modes[0]                 # use first mode
    modes = modes[1:] or [mode]     # trim first, always keep last

    if k_dict is None:              # initiate dictionary
        k_dict = create_k_dict(k_keys, inptdd)

    if len(k_keys) == 0:            # final case, yield copy
        yield OrderedDict((k, v.copy()) for k, v in k_dict.items())

    else:                           # intermediate cases
        for (k_keys_, k_dict_) in set_k_dict_gen(k_keys, k_dict, mode=mode):
            for k_dict in k_dict_gen(k_keys=k_keys_, k_dict=k_dict_, modes=modes):
                yield k_dict  # note: this is one of those copies from above


def create_k_dict(k_keys, inptdd, mode='exclude'):
    ''' Return a k_dict where all the k_vals are set to `mode`. '''
    k_dict = OrderedDict((key, OrderedDict()) for key in k_keys)
    x_key_val_dkey = get_x_dict(inptdd)  # get values
    for k_key in k_dict:
        for k_val in x_key_val_dkey[k_key]:
            k_dict[k_key][k_val] = mode         # set mode
    return k_dict


def set_k_dict_gen(k_keys, k_dict, mode):
    ''' Yield k_keys (reduced) and k_dict (copy, changed). '''

    # Set all values of each key to mode, then yield
    if mode in ['color', 'marker']:             # iterate all keys
        for k_key in k_keys:
            # create a new set of k_keys, without k_key or its dependencies
            k_keys_ = [k for k in k_keys if k not in x_dep[k_key]]
            dep_keys = set(k_keys) - set(k_keys_ + [k_key])

            k_dict_ = OrderedDict(              # create copy
                (k, v.copy())
                for k, v in k_dict.items()
                if k not in dep_keys
                )
            for k_val in k_dict[k_key]:         # set all values
                k_dict_[k_key][k_val] = mode
            yield k_keys_, k_dict_              # then yield all at once

    # Set each value of the first key to mode, then yield
    elif mode == 'include':                # iterate values, once per key
        k_key = k_keys[0]                  # choose first key
        # create a new set of k_keys, without k_key or its dependencies
        k_keys_ = [k for k in k_keys if k not in x_dep[k_key]]
        dep_keys = set(k_keys) - set(k_keys_ + [k_key])

        for k_val in k_dict[k_key]:         # set each value
            k_dict_ = OrderedDict(          # create copy
                (k, v.copy())
                for k, v in k_dict.items()
                if k not in dep_keys
                )
            k_dict_[k_key][k_val] = mode
            yield k_keys_, k_dict_          # and yield each

    else:
        logger.error('Wrong mode: {}'.format(mode))
        raise SystemExit


def get_mode_dkeys(k_dict, inptdd, exclude=True):
    ''' Return a dict with sets of filenames, indexed by their plot mode.

    Parameters
    ----------
    k_dict :    dict
        k_dict, see definitions above
    inptdd :    dict
        filename: simulation input parameters
    exclude :   bool

    Returns
    -------
    md :    dict(mode=set(filenames))
        filenames indexed by their mode.
    '''
    # note: a d_key is a filename
    xkey_xval_dkeys = get_x_dict(inptdd)
    md = {m: set() for m in ['color', 'marker', 'include', 'exclude']}

    for k_key, k_item in k_dict.items():
        for k_val, mode in k_item.items():
            d_keys = xkey_xval_dkeys[k_key][k_val]
            md[mode].update(d_keys)

    md['plot'] = md['color'] | md['marker'] | md['include']

    if exclude:
        for k in ['color', 'marker', 'include', 'plot']:
            md[k] -= md['exclude']

    return md


def get_mode_count(k_dict):
    ''' Return a dict with the number of times each mode is used.'''
    mode_count = {m: 0 for m in ['color', 'marker', 'include', 'exclude']}
    for k_key, k_item in k_dict.items():
        for k_val, mode in k_item.items():
            mode_count[mode] += 1
    return mode_count


def get_mode_count_plotable(k_dict, inptdd):
    ''' Return a dict with the number of times each mode is used.'''
    mode_count = {m: 0 for m in ['color', 'marker', 'include', 'exclude']}
    xkey_xval_dkeys = get_x_dict(inptdd)
    md = get_mode_dkeys(k_dict, inptdd)

    for k_key, k_item in k_dict.items():
        for k_val, mode in k_item.items():
            d_keys = xkey_xval_dkeys[k_key][k_val]
            if (set(d_keys) & md['plot']):
                mode_count[mode] += 1

    return mode_count


def check_k_dict(k_dict, inptdd):
    ''' Return a dict stating how this k-dict can e plotted.

    Modes:
    plotable : At least one data point to plot.
    multi_data : At least two data point to plot.
    multi_color : More than one data point to color.
    multi_marker : More than one data point to mark.
    multi_mc : Minimum two markers and two colors.
    '''

    md = get_mode_dkeys(k_dict, inptdd)
    color = 0
    marker = 0
    xkey_xval_dkeys = get_x_dict(inptdd)
    for k_key, k_item in k_dict.items():
        for k_val, km in k_item.items():
            d_keys = xkey_xval_dkeys[k_key][k_val]
            if set(d_keys) - md['exclude']:
                if (km == 'marker'):
                    marker += 1
                if (km == 'color'):
                    color += 1

    out = {}
    out['all'] = True
    out['unplotable'] = (len(md['plot']) == 0)
    out['plotable'] = (len(md['plot']) > 0)
    out['multi_data'] = (len(md['plot']) > 1)
    out['multi_marker'] = (marker > 1)
    out['multi_color'] = (color > 1)
    out['multi_mc'] = (marker > 1) and (color > 1)
    return out

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       X-DICT TOOLS                                                  #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def validate_x_keys(x_keys=None, warn_invalid=True,
                    default_key='needle_voltage'):
    ''' Return `x-keys` which are in `x-map`. '''

    if x_keys is None:
        x_keys = []
    if type(x_keys) is not list:
        x_keys = [x_keys]

    # map abbreviations to correct keys
    invalid_keys = [x_key for x_key in x_keys if x_key not in x_map]
    valid_keys = [x_map[x_key] for x_key in x_keys if x_key in x_map]
    if warn_invalid and invalid_keys:
        msg = f'Ignoring invalid x-keys: {", ".join(invalid_keys)}'
        logger.warning(msg)

    if default_key and valid_keys == []:
        valid_keys = [x_map['default_key']]
        msg = 'No valid x-keys. Setting x-key to: {default_key}'
        logger.warning(msg)

    logger.debug(f'Returning valid keys: {", ".join(valid_keys)}')
    return valid_keys


def get_x_dict(sim_inputs, mode='all', x_keys=None):
    ''' Return a dict indexed by the parameters,
    then their values, giving a list of filenames.

    Parameters
    ----------
    sim_inputs : dict
                 {filename: sim_input}
    mode :       str
                 Find `all` keys or just the ones with `varied` values.
    x_keys :     lst(str)
                 Return for given keys, overrides `mode`.

    Return
    ------
    x_dict : dict
             {parameter: {value: filename}}
    '''
    x_dict = OrderedDict()        # what you get later if  mode=='all'
    d_keys = sorted(sim_inputs)   # data file keys
    for x_key in x_cfg:
        # create a list of sorted, unique x-values
        def x_get(k):
            try:
                return x_cfg[x_key]['get'](k)
            except KeyError:
                return None
        d_dict = {d_key: x_get(sim_inputs[d_key]) for d_key in d_keys}
        d_dict = {d_key: val
                  for d_key, val in d_dict.items()
                  if val is not None
                  }

        # values should not be lists or sets or dicts, but values
        for key, val in d_dict.items():
            if type(val) is list:
                logger.error('Error! Invalid value (list)')
                logger.info('Perhaps an expandable file is included?')
                return {}

        # sort to ensure same results each time
        x_vals = sorted(set(val for key, val in d_dict.items()))

        # create a dict of d_keys indexed by x_val
        x_dict[x_key] = OrderedDict((x_val, []) for x_val in x_vals)
        for d_key, d_val in d_dict.items():
            x_dict[x_key][d_val].append(d_key)

    # choose/build return dict
    if x_keys is not None:
        # choose only keys specified by x_keys
        x_keys_valid = validate_x_keys(x_keys)
        x_dict_out = OrderedDict(
            (x_key, item)
            for x_key, item in x_dict.items()
            if x_key in x_keys_valid
            )
    elif mode == 'all':
        # choose all keys
        x_dict_out = x_dict
    elif mode == 'varied':
        # choose just keys that are varied
        x_dict_out = OrderedDict(
            (x_key, item)
            for x_key, item in x_dict.items()
            if (len(item) > 1) and (None not in item)
            )
    else:
        logger.error('Invalid mode: {}'.format(mode))
        x_dict_out = {}

    logger.log(5, f'Returning x-dict with keys: {", ".join(x_dict_out)}')
    return x_dict_out


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       PLOT TOOLS                                                    #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calc_statistics(x, y):
    ''' Return unique x-values, with average and std dev for y.

    Parameters
    ----------
    x : float
        x-values
    y : float
        y-values

    Returns
    -------
    xu: float
        x-values, unique
    ya: float
        y-values, average
    ys: float
        y-values, standard deviation
    '''

    # Calculate statistics
    xu = np.unique(x)
    ya = np.zeros(xu.shape)
    ys = np.zeros(xu.shape)
    for i, xi in enumerate(sorted(xu)):
        idx = (x == xi)
        xu[i] = xi
        ys[i] = np.std(y[idx])
        ya[i] = np.average(y[idx])

    return xu, ya, ys


def calc_1d_interpolation(x, y, kind=None, no=50):
    ''' Return linspaced x and interpolated y for average, with pm std dev.

    Parameters
    ----------
    x : float
        x-values
    y : float
        y-values

    Returns
    -------
    xls :   float
            x-values, unique
    ya  :   float
            y-values, average
    yp  :   float
            y-values, average + standard deviation
    ym  :   float
            y-values, average - standard deviation
    '''
    # Calculate and sort variables
    xu, ya, ys = calc_statistics(x, y)
    xls = np.linspace(min(xu), max(xu), num=no)
    yp = ya + ys
    ym = ya - ys

    # 'linear' is safe but not that nice
    # 'quadratic' seems to oscillate
    # 'cubic' seems to oscillate
    # 1 d interpolation
    kinds = [
        'pchip',
        'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
        ]
    if kind is None:
        kind = kinds[0]
    if not len(xu) > 1:
        kind = 'few_xu'
        logger.info(f'Skipping interpolation, too few points: {len(xu)}')

    if kind == 'few_xu':
        ya = np.array([ya for xi in xls])
        yp = np.array([yp for xi in xls])
        ym = np.array([ym for xi in xls])
    elif kind == 'pchip':
        ya = interpolate.pchip_interpolate(xu, ya, xls)
        yp = interpolate.pchip_interpolate(xu, yp, xls)
        ym = interpolate.pchip_interpolate(xu, ym, xls)
    else:
        # Only linear interpolation seems to work well for a small dataset
        ya = interpolate.interp1d(xu, ya, kind=kind)(xls)
        yp = interpolate.interp1d(xu, yp, kind=kind)(xls)
        ym = interpolate.interp1d(xu, ym, kind=kind)(xls)

    return xls, ya, yp, ym


def ax_add_statistics(ax, x, y, desc, to_plot=['a', 'p', 'm']):
    ''' Add average and plus/minus standard deviation to axis. '''
    xu, ya, ys = calc_statistics(x, y)
    yp = ya + ys
    ym = ya - ys
    d = dict(desc, label=None, ls='')

    # Plot statistics
    if 'a' in to_plot:
        d.update({'alpha': 0.8, 'ms': 10})
        ax.plot(xu, ya, **d)
    if 'p' in to_plot:
        d.update({'alpha': 0.3, 'ms': 8})
        ax.plot(xu, yp, **d)
    if 'm' in to_plot:
        d.update({'alpha': 0.3, 'ms': 8})
        ax.plot(xu, ym, **d)


def ax_add_1d_interpolation(ax, x, y, desc, kind=None,
                            to_plot=['a', 'p', 'm'],
                            logx=False, logy=False,
                            ):
    ''' Interpolate data to average and pm std dev, and add to the axis.
    '''
    # 'linear' is safe but not that nice
    # 'quadratic' seems to oscillate
    # 'cubic' seems to oscillate

    # filter values below 0 when plotting log
    remove_mask = np.array(x * 0, dtype=bool)
    if logx:
        remove_mask += (x <= 0)
        x[remove_mask] = None
        x = np.log(x)
    if logy:
        remove_mask += (y <= 0)
        y[remove_mask] = None
        y = np.log(y)
    x = x[~remove_mask]
    y = y[~remove_mask]

    xls, ya, yp, ym = calc_1d_interpolation(x, y, kind=kind)

    if logx:
        xls = np.exp(xls)
    if logy:
        ya = np.exp(ya)
        yp = np.exp(yp)
        ym = np.exp(ym)

    # Plot interpolation
    d = dict(desc, label=None, marker=None)
    if 'a' in to_plot:
        d.update({'ls': ':', 'lw': 2, 'alpha': 0.8})
        ax.plot(xls, ya, **d)
    if 'p' in to_plot:
        d.update({'ls': '--', 'lw': 1, 'alpha': 0.4})
        ax.plot(xls, yp, **d)
    if 'm' in to_plot:
        d.update({'ls': '--', 'lw': 1, 'alpha': 0.4})
        ax.plot(xls, ym, **d)


#
