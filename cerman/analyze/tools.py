#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Tools shared by the module.
'''

# General imports
import numpy as np
from pathlib import Path
import logging

# Import from project files
from .. import tools        # import namespace
from .. tools import *      # import to overwrite
from . import sim_parameters

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

paramr = sim_parameters.paramr          # for representation
paramm = sim_parameters.map_keys        # for map


def get_fpath(data, head=None, folder=None, ext=None, tail=None):
    ''' Returns a path for a plot file.

    Creates the plot folder, if required.

    Parameters
    ----------
    data :  dict
            loaded data
    head :  str
            this is added to the beginning of the file name
    tail :  str
            this is added to the end of the file name
    folder : str
            folder to add plot to
    ext :   str
            plot extension

    Returns
    -------
    str
            file path
    '''

    # Get absolute path to data folder
    if 'header' in data:
        dpath = Path(data['header']['fpath']).resolve()
    else:
        dpath = Path(data['fpath']).resolve()

    return tools.get_fpath(fpath=dpath, folder=folder,
                           head=head, tail=tail, ext=ext)


def padded_no(no, pad=None, no_max=None):
    ''' Add padding to a number.

    Parameters
    ----------
    no    : int
            number to be padded
    pad   : int
            new number width
    no_max : int
            pad up to this many digits

    Returns
    -------
    str
        'no' with padding

    Examples
    --------
    >>> padded_no(3, no_max=543)
    '003'
    >>> padded_no(34, pad=4)
    '0034'
    '''

    b = 1
    if pad is not None:
        b = int(pad)

    if no_max is not None:
        a, b = '{:.2e}'.format(no_max).split('e')
        b = int(b)+1

    return '{no:0{width}d}'.format(no=no, width=b)


def fmt_sci(x, pos):
    ''' Formatter function for exponential notation.
    '''
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)


def get_key_data(loaded_data, keys=None, fmt=None):
    ''' Return a string comprised of the value of each key.

    Parameters
    ----------
    loaded_data :   dict
        loaded simulation data
    keys :  lst(str)
        list of parameter keys to use
    fmt :   str
        format to use for value string

    Returns
    -------
    str
        the value string of each key
    '''
    sim_input = loaded_data['header']['sim_input']

    # get default data
    if keys is None:
        keys = sim_input.get('keys_expanded', ['needle_voltage'])
        if 'random_seed' in keys:
            keys.remove('random_seed')

    # check for unusable keys
    unused_keys = [key for key in keys if key not in list(paramm)]
    keys = [key for key in keys if key in list(paramm)]
    if unused_keys:
        msg = 'Not possible to get data for {} key(s)'
        logger.info(msg.format(len(unused_keys)))
        logger.debug('Keys: {}'.format(', '.join(unused_keys)))

    # map possible abbreviations and get value strings
    keys = [paramm[key] for key in keys]
    msg_lst = [paramr[key].get_val_str(sim_input, fmt=fmt) for key in keys]
    return ', '.join(msg_lst)


# todo: consider moving opt2kwargs and opt2help to top layer tools
def opt2kwargs(options, bools={}, floats={}, strings={}, modify=False):
    '''Return kwargs from a list of options.

    If modify is false, then a warning will be given for unused options.

    Parameters
    ----------
    options : lst(str)
        list of options to be parsed
    bools :     dict
        keywords as keys and explanations as value
    floats :    dict
        keywords as keys and explanations as value
    strings :   dict
        keywords as keys and explanations as value
    modify :    bool
        remove used keys from `options` inplace

    Returns
    -------
    dict
        kwargs
    '''

    logger.log(5, 'Got options: {}'.format(options))
    kwargs = {}    # arguments for plot function
    if options is None:
        options = []
    opts = options.copy()  # to safely remove keys later

    # parse options
    for o in opts.copy():
        if '=' in o:
            k, v = o.split('=', maxsplit=1)

        else:  # assume bool, to be set true
            k = o
            v = True

        # set these true/false based on input on/off
        if (k in bools) and (o in opts):
            if v in [True, 'on', 'true', 'True']:
                kwargs[k] = True
                opts.remove(o)
            elif v in [False, 'off', 'false', 'False']:
                kwargs[k] = False
                opts.remove(o)

        # set floats based on opts
        if (k in floats) and (o in opts):
            kwargs[k] = float(v)    # force error, if needed
            opts.remove(o)

        # set strings based on opts
        if (k in strings) and (o in opts):
            kwargs[k] = str(v)      # force error, if needed
            opts.remove(o)

        if modify and (o not in opts):
            options.remove(o)

    # warn for unused opts
    if opts and not modify:
        msg = 'Warning! {} unused option(s)'.format(len(opts))
        logger.warning(msg)
        logger.info('Unused: ' + ', '.join(opts))

    logger.log(5, 'Returning kwargs: {}'.format(kwargs))
    return kwargs


def opt2help(bools={}, floats={}, strings={}):
    '''Print (log) help text.

    Parameters
    ----------
    bools :     dict
        keywords as keys and explanations as value
    floats :    dict
        keywords as keys and explanations as value
    strings :   dict
        keywords as keys and explanations as value
    '''
    logger.info('')
    logger.info('Available options are listed below.')

    all_opt = {}
    all_opt.update(bools)
    all_opt.update(floats)
    all_opt.update(strings)
    longest = max(len(k) for k in all_opt)

    if bools:
        logger.info('')
        logger.info('Bool options are:')
    for k, v in sorted(bools.items()):
        msg = '{k:>{lt}} : {v}'.format(k=k, v=v, lt=longest)
        logger.info(msg)

    if floats:
        logger.info('')
        logger.info('Float options are:')
    for k, v in sorted(floats.items()):
        msg = '{k:>{lt}} : {v}'.format(k=k, v=v, lt=longest)
        logger.info(msg)

    if strings:
        logger.info('')
        logger.info('String options are:')
    for k, v in sorted(strings.items()):
        msg = '{k:>{lt}} : {v}'.format(k=k, v=v, lt=longest)
        logger.info(msg)

    logger.info('')


#
