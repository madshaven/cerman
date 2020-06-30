#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Tools related to simulation input parameters.
'''

# general imports
from collections import OrderedDict
import logging
import numpy as np
from pathlib import Path

# imports from package
from .. import tools
from .calc_missing import calc_missing_params
from .defaults import get_defaults

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def read_input_file(fpath):
    ''' Read input file and return simulation parameters.

    Update the default parameters with the data from the input file.
    Calculate parameters as needed.
    Add information about current directory.
    Add information about how different keys are treated.
    Dump the resulting input parameters to a file, if requested.

    Parameters
    ----------
    fpath : str
            file path

    Returns
    -------
    params_out :  dict
                  parameters for simulation
    '''

    fpath = Path(fpath)
    logger.debug(f'Reading input from "{fpath.name}"')

    # Break is no file is present
    if not fpath.is_file():
        msg = f'Error, "{str(fpath)}" is not an input file!'
        logger.error(msg)
        raise SystemExit()

    # Load parameters and check how they differs from default parameters
    try:
        params_loaded = tools.json_load(fpath)

    except BaseException as ex:
        msg = 'Failed loading {}'.format(str(fpath))
        logger.error(msg)

        ex_msg = 'An exception of type {0} occurred.\nArguments: {1!r}'
        logger.debug(ex_msg.format(type(ex).__name__, ex.args))
        logger.log(5, 'Stack trace:', exc_info=True)

        raise SystemExit()

    params_default = get_defaults()
    # Note! Only top level keys
    keys_default = [key for key in params_default]    # keep default sorting
    keys_loaded = [key for key in keys_default if (key in params_loaded)]
    keys_missing = [key for key in keys_default if (key not in keys_loaded)]
    keys_extra = [key for key in keys_loaded if (key not in params_default)]
    keys_null = [key for key, val in params_loaded.items() if (val is None)]

    # Null/none keys to be replaced by default values!
    for key in keys_null:
        params_loaded.pop(key, None)

    # Build as copy of defaults, then update with loaded parameters
    params_out = params_default.copy()

    def update_r(d, u):
        ''' Recursively update dict `d` with dict `u`.'''
        for key, val in u.items():
            if isinstance(val, dict):
                r = update_r(d.get(key, {}), val)
                d[key] = r
            elif u[key] is not None:    # do not update with `None`
                d[key] = u[key]
        return d

    # Update dict (recursive)
    params_out = update_r(params_out, params_loaded)

    # Calculate missing parameters and add them
    data_calculated = calc_missing_params(params_out)
    keys_calculated = sorted(data_calculated)
    params_out.update(data_calculated)

    # Set project working directory
    params_out['pwd']   = str(fpath.resolve().parent)
    params_out['name']  = str(fpath.stem)
    params_out['fpath'] = str(fpath.resolve())

    # Add some info
    params_out['keys_loaded']     = keys_loaded
    params_out['keys_default']    = keys_default
    params_out['keys_missing']    = keys_missing
    params_out['keys_extra']      = keys_extra
    params_out['keys_calculated'] = keys_calculated
    params_out['keys_null'] = keys_null

    if keys_extra:
        msg = 'WARNING! Found {} extra key(s):'.format(len(keys_extra))
        logger.warning(msg)
        logger.info(str(keys_extra))

    # Save input
    if params_out['save_input']:
        out_path = str(fpath.resolve().with_suffix('.input'))
        tools.json_dump(params_out, out_path)

    # Return data to be used as input parameters to a simulation
    return params_out


def create_input_files(fpath):
    ''' Read file, expand lists, and dump files.

    Parameters
    ----------
    fpath : str
            file path to expandable file

    Returns
    -------
    fpaths : lst[str]
             paths to created files
    '''

    # todo: deal with 'none', important for sublists
    # idea: consider removing 'none' from input files
    # idea: custom save_dicts to be true by default, or warn!

    fpath = Path(fpath)

    # Verify that file exists
    if not (fpath.is_file()):
        msg = 'The file "{}" does not exist.'.format(fpath)
        logger.error(msg)
        raise SystemExit()

    # Load data
    try:
        params_loaded = tools.json_load(fpath)
        params_default = get_defaults()

    except BaseException as ex:
        logger.info('Loading "{}" failed.'.format(str(fpath)))
        logger.info('Perhaps you forgot a comma at the end of a list?')
        logger.info('''Or used ' instead of " ?''')

        ex_msg = 'An exception of type {0} occurred.\nArguments: {1!r}'
        logger.debug(ex_msg.format(type(ex).__name__, ex.args))
        logger.log(5, 'Stack trace:', exc_info=True)

        return []

    params = params_loaded.copy()

    # Add all defaults to params
    if params.get('add_defaults', params_default['add_defaults']):
        # use recursive updating of dict
        params = tools.update_dict(params_default.copy(), params)
        logger.log(5, 'Added default params')

    # Add default params, if required
    keys = ['seq_start_no', 'exp_folder', 'add_defaults',
            'simulation_runs', 'random_seed', 'permutation_order']
    for key in keys:
        if key not in params:
            params[key] = params_default[key]

    # get keys for values that are strings
    # assume 'linspace(start, stop, num)'
    keys_s = [k for k, v in params.items() if (type(v) is str)]
    keys_ls = [k for k in keys_s if (params[k].startswith('linspace'))]
    logger.debug(f'Expanding linspaces from {keys_ls}')
    for key in keys_ls:
        string = params[key]
        values = string[len('linspace('):-1].split(',')
        start = float(values[0])
        stop = float(values[1])
        num = int(values[2])
        params[key] = np.linspace(start, stop, num, endpoint=True).tolist()

    # Change from a dict with some lists, to a dict of lists
    for k, v in params.items():
        if type(v) != list:
            params[k] = [v]

    # Expand random seed
    params['random_seed'] = [
        (s + i)
        if (s is not None) else None  # keep any None
        for s in params['random_seed']
        for i in range(params['simulation_runs'][0])
        ]
    logger.log(5, 'Expanded random seed to ' + str(params['random_seed']))

    keys_expanded = [
        key for key, val in params.items()
        if (len(val) > 1) and (key in params_default)  # expand only usable keys
        ]
    if keys_expanded:
        msg = 'Found {} key(s) to expand: '.format(len(keys_expanded))
        logger.info(msg + ', '.join(keys_expanded))
    else:
        logger.warning('Found no keys to expand.')

    keys_unused = [key for key in params if key not in params_default]
    if keys_unused:
        msg = f'Warning! Removed {len(keys_unused)} unused key(s): '
        logger.warning(msg + ', '.join(keys_unused))
        [params.pop(key) for key in keys_unused]

    # Sort keys as desired for expansion
    keys = [k for k in params_default if k in params]
    first_keys = [k.strip()
                  # note: everything is a list at this point
                  for k in params['permutation_order'][0].split(',')
                  if k.strip()]  # remove empty keys
    logger.debug(f'First permutated keys are {first_keys}')
    _invalid_keys = [k for k in first_keys if k not in keys]
    if _invalid_keys:
        first_keys = [k for k in first_keys if k not in _invalid_keys]
        logger.warning(f'Removed keys from permutation order: {_invalid_keys}')
    keys = first_keys + [k for k in keys if k not in first_keys]

    # Permute! Populate all lists to full length
    m = 1
    s = 1  # final length of all lists
    for key in keys:
        s *= len(params[key])
    for key in keys:
        l = len(params[key])
        k = int(s / (l * m))
        params[key] = [_l for _k in range(k)
                        for _l in params[key]
                        for _m in range(m)]
        m *= l

    # change `None` to a random seed
    np.random.seed(seed=None)  # initialize from entropy
    params['random_seed'] = [
        np.random.randint(0, int(2**32))  # [0, 2**32>
        if s is None
        else
        s
        for s in params['random_seed']
        ]
    logger.log(5, f'Changed random seed to {params["random_seed"]}')

    # Change from dict to list
    params = tools.dict_of_list_to_list_of_dicts(params)

    # set format of sequence number
    seq_start_no = params[0]['seq_start_no']
    digits = max(3, len(str(seq_start_no + len(params) - 1)))
    fmt = f'0{digits}d'

    # Dump files
    fpaths = []
    for i, params_out in enumerate(params):

        params_out['keys_expanded'] = keys_expanded
        params_out['keys_unused'] = keys_unused

        seq_no = params_out['seq_start_no'] + i
        folder = params_out['exp_folder']
        name_i = f'{fpath.stem}_{seq_no:{fmt}}'
        fdir_i = fpath.parent / folder

        if not fdir_i.is_dir():
            fdir_i.mkdir()

        fpath_i = str(fdir_i / name_i) + '.json'
        fpaths.append(fpath_i)

        tools.json_dump(params_out, fpath_i)

        logger.debug('Created {}'.format(str(fpath_i)))

    logger.info('Created {} files.'.format(len(params)))
    return fpaths


def dump_defaults(fpath=None):
    ''' Dump simulation default parameters to `fpath`.'''

    # Load default parameters
    parameters = get_defaults()

    if fpath is None:
        fpath = parameters['fpath']
    fpath = Path(fpath)

    # Dump to file
    try:
        tools.json_dump(parameters, fpath)
        logger.info(f'Dumped default parameters to {fpath}.')
    except:
        msg = f'Could not dump data to: {fpath}.'
        logger.error(msg)
        msg = 'Please create the folder, '
        msg += 'or change the default directory to an existing folder.'
        logger.info(msg)
        logger.log(5, 'Stack trace:', exc_info=True)
        raise SystemExit()

    return fpath


#
