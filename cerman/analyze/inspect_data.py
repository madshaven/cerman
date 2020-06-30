#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This module contains tools for inspecting saved simulation data.

    This module to inspect the content,
    and print to screen and/or file.
    The `load_data` module is used to load the save data.
'''

# General imports
import pickle
import logging
import sys

# Import from project files
from . import tools
from .load_data import LoadedData

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def inspect_data(fpath, glob=None, idxes=None, options=None):
    ''' Inspect saved data and print properties.

    Parameters
    ----------
    fpath : str
            file or folder name
    options : str
              type `help` to see options
    glob  : str
            globbing pattern for folder
    '''

    fpaths = tools.glob_fpath(fpath, glob=glob, post='.pkl')

    opt_bool = {
        'meta': 'Show data keys and length (default)',
        'type': 'Show data types',
        'size': 'Calculate size of each key',
        'details': 'Show details of each key',
        'header': 'Print file header',
        'footer': 'Print file footer',
        }
    opt_string = {
        'keys': 'Show for given keys. Defaults to all.',
        'no_range': 'Iteration numbers for details `no_range=from_to`.',
        }

    if options is None:
        options = []

    if ('help' in options) or ('info' in options):
        tools.opt2help(
            bools=opt_bool,
            strings=opt_string,
            )
        return

    kwargs = tools.opt2kwargs(
        options=options,
        bools=opt_bool,
        strings=opt_string,
        )

    # set meta as true by default
    if 'meta' not in kwargs:
        kwargs['meta'] = True

    keys = kwargs.get('keys', None)
    if keys is not None:
        keys = keys.split(',')
    logger.log(5, f'Keys: {keys}')

    def print_func(fpath):
        data = LoadedData(fpath).data       # load data

        if kwargs.get('meta'):
            inspect_meta(data)
        if kwargs.get('header'):
            inspect_header(data)
        if kwargs.get('footer'):
            inspect_footer(data)
        if kwargs.get('type'):
            inspect_type(data, keys=keys)
        if kwargs.get('size'):
            inspect_size(data, keys=keys)
        if kwargs.get('details'):
            no_range = kwargs.get('no_range', None)
            inspect_details(data, keys=keys, no_range=no_range, idxes=idxes)
        logger.info('')
        data = {}                               # free memory

    info = 'Printing data from {fpath}'
    tools.map_fpaths(fpaths, func=print_func, info=info)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       PRINT DATA                                                    #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def _validate_keys(datadict, keys):
    ''' Return keys that can be used. Defaults to all.
    '''
    # get all keys
    logger.log(5, f'Got keys: {keys}')
    all_keys = [k for k in sorted(datadict) if k not in ['header', 'footer']]
    # set default
    if (keys is None) or (keys == '') or (keys == 'all') or ('all' in keys):
        keys = all_keys
    if type(keys) is str:
        keys = [keys]

    # remove invalid keys
    wrong_keys = [k for k in keys
                  if ((k not in datadict) or (k in ['header', 'footer']))
                  ]
    if wrong_keys:
        logger.warning(f'Removing keys not in data: {", ".join(wrong_keys)}')
    keys = [k for k in keys if k not in wrong_keys]

    logger.log(5, f'Using keys: {keys}')
    return keys


def _hf_dict2str(dictionary):
    msg = ''
    for k, v in sorted(dictionary.items()):
        msg += f'{k:}:\n{v}\n'
    return msg


def inspect_header(data):
    msg = _hf_dict2str(data['header'])
    logger.info(f'### HEADER ###\n{msg}')


def inspect_footer(data):
    msg = _hf_dict2str(data['footer'])
    logger.info(f'### FOOTER ###\n{msg}')


def inspect_meta(data):
    ''' Print information about the length of the data file and its keys. '''
    msg = '### META ###\n'
    msg += f'Data from {data["header"]["fpath"]}.\n'
    msg += f'Length of `no` is {len(data["no"])} and '
    msg += f'data is containing {len(data)} keys:\n'
    msg += f'{", ".join(sorted(data))}\n'

    logger.info(msg)
    return msg


def inspect_size(datadict, keys=None, protocol='pickle'):
    ''' Print space requirements for dumping a dictionary.

    Note: logger.info is used to print.

    Note: `marshal` do not work as well as `pickle`.
    Some data structures may be corrupted when using `marshal`.

    Parameters
    ----------
    datadict :  dict
                dictionary to inspect
    protocol :  str
                protocol to use for dumps

    Returns
    -------
    msg :   str
        The the logged message.
    '''

    keys = _validate_keys(datadict, keys)

    if protocol=='pickle':
        # import pickle
        def dumps(data):
            return pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
    # elif protocol=='marshal':
    #     import marshal
    #     def dumps(data):
    #         return marshal.dumps(data)
    else:
        logger.error(f'Unsupported protocol: {protocol}')
        return ''

    # get size of whole dict
    string = dumps(datadict)
    dict_size = sys.getsizeof(string) / 1e6

    # specify line formats
    fmt1 = ' {: >7}  {: >6}  {}\n'
    fmt2 = ' {: >7.3f}  {: >6.2f}  {}\n'
    line = fmt1.format('-'*7, '-'*6, '-'*5)

    # build message - header then each key
    msg = '### DATA SIZE ###\n'
    msg += f'Using {protocol} for dumps.\n\n'
    msg += fmt1.format('[MB]', '[%]', 'Key')
    msg += line
    msg += fmt2.format(dict_size, 100, 'Total')
    msg += line
    for key in keys:
        string = dumps(datadict[key])
        abs_size = sys.getsizeof(string) / 1e6
        rel_size = abs_size / dict_size
        msg += fmt2.format(abs_size, rel_size * 100, key)
    msg += line

    # log and return
    logger.info(msg)
    return msg


def inspect_type(datadict, keys=None):
    ''' Print data key names, length and data type for given keys.

    Note: logger.info is used to print.

    Parameters
    ----------
    datadict : dict
            data dictionary to print
    keys  : lst[str], optional
            Print only these keys, if specified
    info  : bool, optional
            Print key length and type

    Returns
    -------
    msg :   str
        The the logged message.
    '''

    keys = _validate_keys(datadict, keys)

    # find max length of keys, data length, and data type
    l_types = max(len(str(type(datadict[k][0]))) for k in keys)
    l_keys = max(len(k) for k in keys)
    l_ints = max(len(str(len(datadict[k]))) for k in keys)
    l_ints = max(l_ints, len('length'))

    # specify line formats
    fmt1 = '{{:>{0}}}  {{:{1}}}  {{:{2}}}\n'
    fmt1 = fmt1.format(l_keys, l_ints, l_types)
    logger.log(5, 'fmt1: {}'.format(fmt1))
    line = fmt1.format('-' * l_keys, '-' * l_ints, '-' * l_types)

    # build output
    msg = '### DATA TYPE ###\n'
    msg += line
    msg += fmt1.format('Key', 'length', 'type')
    msg += line
    for key in keys:
        msg += fmt1.format(key, len(datadict[key]), str(type(datadict[key][0])))
    msg += line

    # log and return
    logger.info(msg)
    return msg


def inspect_details(datadict, keys=None, no_range=None, idxes=None, nos=None):
    ''' Inspect data. Print result to screen.

    Parameters
    ----------
    datadict  : dict
            data dictionary to print
    keys  : lst[str], optional
            Print only these keys, if specified
    no_range : tup or str
            range of iteration numbers to use (start, stop) or `start_stop`
    nos     : lst
            iteration numbers to use
    idxes : lst
            list of indecies in saved data to use
    info  : bool, optional
            Print key length and type
    '''

    # set default keys
    keys = _validate_keys(datadict, keys)


    #
    if nos is None:
        nos = []
    logger.log(5, f'nos at start: {nos}')

    if type(no_range) is str:
        start, stop = no_range.split('_')
        no_range = (int(start), int(stop))
    if type(no_range) is tuple:
        nos += list(range(*no_range))
        logger.log(5, f'nos after no_range: {nos}')

    if idxes is not None:
        idxes = [idx for idx in idxes
                 if (idx >= 0 and idx < len(datadict['no']))
                 ]
        nos += [datadict['no'][idx] for idx in idxes]
        logger.log(5, f'nos after idxes: {nos}')

    # Chose everything is nothing is given
    if len(nos) == 0:
        nos = datadict['no']

    # Pick out only valid no's, and convert to indexes
    nos = [no for no in sorted(set(nos)) if no in datadict['no']]
    idxes = [datadict['no'].index(no) for no in nos]

    logger.debug(f'Found {len(nos)} valid numbers')
    logger.log(5, f'Valid nos: {str(nos)}')

    # Print the data
    msg = '### DETAILS ###\n'
    msg += f'Data from {datadict["header"]["fpath"]}\n'
    msg += f'Printing the data for {", ".join(keys)}\n\n'
    out_dict = {}
    for key in keys:
        data_list = [datadict[key][idx] for idx in idxes]
        out_dict[key] = data_list
        msg += f'{key:s}: \n{data_list}\n'

    # log and return
    logger.info(msg)
    return msg


#
