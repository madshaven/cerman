#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This module contains tools for loading saved data.

    Use the class `LoadedData` to load a saved file.
    The class property `data` contains a dict with a key for
    the `header` and each of the `save_key`s.
    The header contains the keys: time_now, file_type, file_info,
    save_spec_to_dict, sim_input.
'''

# General imports
import pickle
from pathlib import Path
import logging
import numpy as np
import operator
from collections import defaultdict, OrderedDict

# import from project files
from . import tools
from ..core.eh_list import EHList

# settings
eps = np.finfo(float).eps  # 2.22e-16 for double
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

operators = {
    "<":  operator.lt,
    "lt": operator.lt,
    "<=": operator.le,
    "le": operator.le,
    ">":  operator.gt,
    "gt": operator.gt,
    ">=": operator.ge,
    "ge": operator.ge,
    "==": operator.eq,
    "eq": operator.eq,
    "!=": operator.ne,
    "ne": operator.ne,
    }

''' ideas:
    - consider http://msgpack.org/
    - consider plain numpy.savez
    - consider a change to pandas for data (no room for meta data?)
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       LOADED DATA                                                   #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class LoadedData():
    ''' A class containing data loaded from saved simulation data.

    Create the class instance either from file, or by adding data explicitly.

    The main properties are:
    'sim_input', 'data', 'header', 'footer'.
    '''

    def __init__(self, fpath=None, data=None, header_only=False):
        ''' Load a file and parse the data, or just parse the data given. '''
        self.fpath = None  # set when loading or parsing data
        if fpath is not None:
            self.load_data(fpath, header_only=header_only)
        elif data is not None:
            self._loaded_data = data
            self._parse_loaded_data()
        else:
            logger.error('Cannot create class without data.')
            return None

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       DATA MANAGEMENT                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def load_data(self, fpath, header_only=False):
        ''' Load and parse data from file. '''
        self.fpath = fpath
        self._loaded_data = load_data(fpath, header_only=header_only)
        self._parse_loaded_data()

    def _parse_loaded_data(self):
        self._header = self._loaded_data['header']
        if 'footer' in self._loaded_data:
            self._footer = self._loaded_data['footer']
        else:
            self._footer = {}
        # self._loaded_data.pop('header')
        # self._loaded_data.pop('footer')

        self._sim_input = self._header['sim_input']
        if not self.fpath:
            self.fpath = self._header['fpath']

    def remove_keys(self, keys):
        # idea: add option to remove keys, useful for saving a subset.
        raise NotImplementedError

    def save_data(self, fpath=None, keys_keep=None, keys_remove=None):
        # idea: add option to save data, possibly as a subset
        raise NotImplementedError

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       PROPERIES                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # note: these are dicts and can be manipulated in place

    # all the saved data in one batch
    # the data-dict contains all the save_keys + header and footer
    data = property(lambda self: self._loaded_data)

    # the header of the loaded save file
    header = property(lambda self: self._header)

    # the footer of the loaded save file
    footer = property(lambda self: self._footer)

    # the simulation input parameters of the loaded save file
    sim_input = property(lambda self: self._sim_input)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       GETTING DATA                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get(self, key, idx=None):
        ''' idea: Try to get key, or return None, but give a notification. '''
        raise NotImplementedError

    def get_nos(self, nos=None, idxes=None):
        ''' Return valid iteration numbers from lists of no's and indexes.
            Default to returning all.
        '''
        nos = nos or []
        idxes = idxes or []
        # pick out idxes in correct range, and make nos list
        _idxes = [idx for idx in idxes
                 if (idx >= 0 and idx < len(self.data['no']))
                 ]
        idx_nos = [self.data['no'][idx] for idx in _idxes]

        # combine lists and then chose everything if nothing is given
        nos = nos + idx_nos
        if not nos:
            nos = self.data['no']

        # Pick out only valid no's and return them
        nos = [no for no in sorted(set(nos)) if no in self.data['no']]
        logger.debug(f'Found {len(nos)} valid numbers')
        logger.log(5, f'{nos}')
        return nos

    def get_heads(self, which=None, idx=None, no=None):
        ''' Reconstruct heads and return as EHList. '''
        return get_heads(self._loaded_data, which=which, idx=idx, no=no)

    def get_seeds(self, key=None, idx=None, no=None):
        ''' Reconstruct seeds property from data. '''
        return get_seeds(self._loaded_data, key=key, idx=idx, no=no)

    def find_first_incidence(self, key, val=0, op='gt'):
        ''' Return first index and iteration number matching a criterion. '''
        return find_first_incidence(self._loaded_data, key, val=val, op=op)

    def find_first_avalanche(self):
        ''' Return index of first avalanche. '''
        return find_first_avalanche(self._loaded_data)

    def find_first_added_avalanche(self):
        ''' Return index of first added avalanche. '''
        return find_first_added_avalanche(self._loaded_data)

    def find_first_z_change(self):
        ''' Return index of first change of z. '''
        return find_first_z_change(self._loaded_data)

    def calc_average_speed(self, k0=0, k1=1):
        return calc_average_speed(self._loaded_data, k0=k0, k1=k1)

    def calc_gap_fraction(self):
        return calc_gap_fraction(self._loaded_data)

    def inception_occurred(self):
        ''' Return True if inception has occurred. '''
        return inception_occurred(self._loaded_data)

    def breakdown_occurred(self):
        ''' Return True if breakdown has occurred. '''
        return breakdown_occurred(self._loaded_data)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       LOAD DATA                                                     #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_data(fpath, save_processed=False, header_only=False):
    ''' Load data from a pickle-file.

    Parameters
    ----------
    fpath : str or Path
            file or folder path
    save_processed : bool
            save save file, overwriting as one chunk, less space, more memory

    Returns
    -------
    dict
        The returned data is a dict of lists.
        (Converted from the data loaded from disk, which is a series of dicts.)

    '''
    # idea: add progress bar, or a tick for every n'th block loaded
    #       count time for the first 10, then consider blocks?

    # Check for correct input
    fpath = Path(fpath)
    if not fpath.is_file():
        raise FileNotFoundError(f'The file {fpath} does not exist.')

    # Load header
    try:
        header = next(tools.pickle_load_gen(fpath))
    except:
        logger.error(f'Could not read pickle data from {fpath}')
        raise

    if 'file_type' not in header:
        logger.error(f'The header is not correct. File: {fpath}')
        raise SystemExit()

    if header_only:
        header['fpath'] = str(fpath)
        return dict(header=header)

    # Load data
    if header['file_type'] == 'save_raw':
        data = load_raw_data(fpath)
    elif header['file_type'] == 'save_processed':
        data = load_processed_data(fpath)
    else:
        msg = 'Could not find correct procedure for loading file.'
        logger.error(msg)
        raise SystemExit()

    # Add fpath
    data['header']['fpath'] = str(fpath)

    # Save processed data
    # note: pickle requires a lot of memory, about twice of the dump size
    #       data should therefore be dumped in reasonable batches
    if (header['file_type'] == 'save_raw') and save_processed:
        logger.warning('Save processed is not applied.')
        # save_processed_data(data, fpath=fpath)

    return data


def load_raw_data(fpath):
    ''' Load data from a pickle-file.

    Expected file structure: header, single_data, ... single_data, footer

    Parameters
    ----------
    fpath : str or Path
            file or folder path

    Returns
    -------
    dict
        The returned data is a dict of lists.
        (Converted from the data loaded from disk, which is a series of dicts.)

    '''
    # Consider adding progress.
    # It is easy to add a bar for each block loaded, but how to count them in advance?
    # Display progress if initial load takes more than dt time?
    # Add a looping progress bar for each load? ### --> #### ?

    data = defaultdict(list)    # enables appending without initiating
    data['header'] = {}         # leave empty if no header
    data['footer'] = {}         # leave empty if no footer
    keys = []                   # header, data keys, footer
    loader = tools.pickle_load_gen(fpath)

    # Load data
    try:
        # Process rest of the file
        for data_tmp in loader:
            if 'file_info' in data_tmp:         # header
                data['header'] = data_tmp
            elif 'footer_comment' in data_tmp:  # footer
                data['footer'] = data_tmp
            else:                               # data
                if not keys:                    # save key order
                    keys = [k for k in data_tmp]
                for key in data_tmp:            # append data
                    data[key].append(data_tmp[key])
            data_tmp = None                     # clear to save memory

    except:
        logger.error('Could not read pickle data from {fpath}')
        raise

    keys = ['header'] + keys + ['footer']
    return OrderedDict((k, data.pop(k)) for k in keys)


def load_processed_data(fpath):
    ''' Load data from a pickle-file.

    Expected file structure: header, all_data, footer

    Parameters
    ----------
    fpath : str or Path
            file or folder path

    Returns
    -------
    dict
        The returned data is a dict of lists.
    '''

    data = defaultdict(list)
    loader = tools.pickle_load_gen(fpath)

    # Load data
    try:
        data['header'] = next(loader)
        data = next(loader)
        data['footer'] = next(loader)

    except:
        logger.error(f'Could not read pickle data from {fpath}')
        raise

    return data


def load_serial_data(fpath):
    ''' Load data from a pickle-file.

    Expected file format:
        - header : dict
        - data : {'key': key, 'val': val}
        - data : {'key': key, 'val': val}
        - data : {'key': key, 'val': val}

    Parameters
    ----------
    fpath : str or Path
            file or folder path

    Returns
    -------
    dict
        The returned data is a dict of lists.
    '''

    data = OrderedDict()
    loader = tools.pickle_load_gen(fpath)

    try:
        for data_tmp in loader:
            if 'file_info' in data_tmp:         # header
                data['header'] = data_tmp
            elif 'footer_comment' in data_tmp:  # footer
                data['footer'] = data_tmp
            else:                               # data
                data[data_tmp['key']] = data_tmp['val']
            data_tmp = None                     # clear to save memory

    except:
        logger.error(f'Could not read pickle data from {fpath}')
        raise

    return data


def save_processed_data(data, fpath=None):
    ''' Dump processed data.
    (This takes about 3-4x as long as loading raw...)
    '''
    if fpath is None:
        fpath = Path(data['header']['fpath']).with_suffix('.pkl')
    else:
        fpath = Path(fpath)

    data['header']['file_type'] = 'save_processed'

    # write to temporary file
    tpath = fpath.with_suffix('.ptmp')
    with tpath.open('wb') as f:
        # Dump header
        pickle.dump(data['header'], f, pickle.HIGHEST_PROTOCOL)
        # Dump everything
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    # replace original file
    # tpath.replace(fpath)


def save_serial_data(data, fpath=None):
    ''' Dump serial data.
    '''
    if fpath is None:
        fpath = Path(data['header']['fpath']).with_suffix('.pkl')
    else:
        fpath = Path(fpath)

    data['header']['file_type'] = 'save_serial'
    keys = [key for key in sorted(data) if key not in ['header', 'footer']]

    # write to temporary file
    tpath = fpath.with_suffix('.stmp')
    with tpath.open('wb') as f:
        # Dump header
        pickle.dump(data['header'], f, pickle.HIGHEST_PROTOCOL)
        # Dump data
        for key in keys:
            data_tmp = {'key': key, 'val': data[key]}
            pickle.dump(data_tmp, f, pickle.HIGHEST_PROTOCOL)

    # replace original file
    # tpath.replace(fpath)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       GET DATA                                                      #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_sim_input(data):
    ''' Return `sim_input`.

    Parameters
    ----------
    data  : dict
            data dictionary

    Returns
    -------
    sim_input : dict
    '''

    return data['header']['sim_input']


def get_heads(data, which=None, idx=None, no=None):
    ''' Analyze data to reconstruct heads.

    Parameters
    ----------
    data  : dict
            data dictionary
    which : str
            Heads to use, `previous`, [`current`], `removed`, `appended`
    idx   : int
            data idx to get, [`0`]
    no    : int
            data iteration no to plot, overrides 'idx' if specified

    Returns
    -------
    EHList
    '''

    idx = 0 if idx is None else idx
    which = 'current' if which is None else which

    # Override 'idx' if 'no' is specified
    if no is not None:
        idx = data['no'].index(no)
    if idx == -1:
        idx = len(data['no']) - 1

    logger.log(5, f'Getting `{which}` heads at index `{idx}`')

    # get the correct heads
    if which == 'current':
        ehdl = data['streamer_heads_dict'][idx]
        ehl = EHList.from_dict_list(ehdl)

    elif which == 'previous':
        ehdl = data['streamer_heads_dict'][idx]
        ehdla = data['streamer_heads_dict_appended'][idx]
        ehdlr = data['streamer_heads_dict_removed'][idx]

        # combine current and removed heads, but none appended heads
        ehdlp = []
        for h in ehdl + ehdlr:
            match = [np.allclose(_h['pos'], h['pos']) for _h in ehdla]
            if not any(match):
                ehdlp.append(h)

        ehl = EHList.from_dict_list(ehdlp)

    elif which == 'removed':
        ehdlr = data['streamer_heads_dict_removed'][idx]
        ehl = EHList.from_dict_list(ehdlr)

    elif which == 'appended':
        ehdla = data['streamer_heads_dict_appended'][idx]
        ehl = EHList.from_dict_list(ehdla)

    else:
        logger.error(f'Error, invalid value for heads ({which})')

    logger.log(5, f'Returning {len(ehl)} heads')
    return ehl


def get_seeds(data, key=None, idx=None, no=None):
    ''' Reconstruct seeds property from data.

    See `sim_data` for valid keys for `seeds`, e.g. `seeds_pos_to_remove`.

    Properties to get: 'no', 'pos', 'Q', 'dQ'.
    Possible conditions or types of seeds:
    'ion', 'electron', 'avalanche', 'critical',
    'behind_roi', 'in_streamer', 'to_append', 'to_remove'.

    Parameters
    ----------
    data  : dict
            data dictionary
    key   : str
            property to get, see `sim_data` for all `seed` keys
    idx   : int
            data idx to get
    no    : int
            data iteration no to plot, overrides 'idx' if specified

    Returns
    -------
    pos  : array
        seed positions or seed property
    '''


    key = key or 'seeds_pos'
    if no is not None:  # override 'idx' if 'no' is specified
        idx = data['no'].index(no)
    elif idx == -1:
        idx = len(data['no']) - 1
    elif idx - 1 > len(data['no']):
        logger.error('Invalid index given.')
        return np.array([])

    # split the given key
    ks = key.split('_')         # split
    assert (ks[0] == 'seeds'), "ks[0] != 'seeds'"
    # pick out the 'property' part
    kp = ks[1]                  # property
    assert (kp in ['no', 'pos', 'Q', 'dQ']), "kp in ['no', 'pos', 'Q', 'dQ']"
    # find the 'condition', or 'type' of seed
    kc = '_'.join(ks[2:])       # condition
    assert kc in ['ion', 'electron', 'avalanche', 'critical',
                  'behind_roi', 'in_streamer',
                  'to_append', 'to_remove', ''], 'kc not correct'
                  # note: the final empty string is for getting all

    # construct the proper key for the property to get
    kp = 'seeds_' + kp
    # construct the proper key for the condition/type of seed to get
    kc = 'seeds_is_' + kc

    logger.log(5, 'ks: ' + ', '.join(ks))
    logger.log(5, 'kp: ' + kp)
    logger.log(5, 'kc: ' + kc)
    logger.log(5, 'key: ' + key)
    # logger.log(5, 'dks: ' + ', '.join(sorted(data)))

    # if the key is in the data, just return it
    if key in data:
        return data[key][idx]

    # if a number is asked for, it can be reconstructed from any bool
    elif (kp == 'no') and (kc in data):
        c = data[kc][idx]
        return data[kc][idx].sum()

    # pos, Q, and dQ can be reconstructed if the condition is in the data
    elif (kp in data) and (kc in data):
        p = data[kp][idx]
        c = data[kc][idx]
        if kp == 'seeds_pos':
            return p[:, c]
        elif kp in ['seeds_Q', 'seeds_dQ']:
            return p[c]

    else:
        logger.warning(f'Could not get seeds for key ({key}).')
        logger.log(5, f'key: {key}')
        logger.log(5, f'kp: {kp}')
        logger.log(5, f'ks: {ks}')
        return np.array([])


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       FIND IDX AND NO                                               #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def find_first_incidence(data, key, val=0, op='gt'):
    ''' Return first index and iteration number matching a criterion.
    The criterion is given by (data[key][idx] op val).

    If the data is an ndarray, then its size will be used.

    Parameters
    ----------
    data  : dict
            iteration data
    key   : str
            data key to consider
    val   : float
            comparison value
    op    : str
            logical operator to use

    Returns
    -------
    idx : int
          index of first iteration matching the criterium
    no  : int
          first iteration number matching the criterium
    '''
    if key not in data:
        logger.info(f'Cannot find incidence for key not in data: {key}.')
        return None, None

    no = -1
    idx = -1

    items = data.get(key, [])
    if type(items[0]) == np.ndarray:
        items = [item.size for item in items]

    for i, item in enumerate(items):
        if operators[op](item, val):
            idx = i
            no = data.get(data['no'][idx])
            break

    return idx, no


def find_first_avalanche(data):
    ''' Return index of first avalanche.
    Parameters
    ----------
    data  : dict
            iteration data
    '''
    possible_keys = [
        'seeds_cum_critical',
        'seeds_no_critical',
        'seeds_Q_max',
        ]
    idx, no = find_first_incidence(data, key=possible_keys[1])
    return idx


def find_first_added_avalanche(data):
    ''' Return index of first added avalanche.

    Parameters
    ----------
    data  : dict
            iteration data
    '''
    possible_keys = [  # Keys that will be greater than zero
        'streamer_pos_appended',
        'streamer_no_appended',
        'streamer_cum_appended',
        ]
    idx, no = find_first_incidence(data, key=possible_keys[0])

    return idx


def find_first_z_change(data):
    ''' Return index of first change of z.

    Parameters
    ----------
    data  : dict
            iteration data
    '''
    idx, no = find_first_incidence(
        data,
        key='streamer_z_min',
        val=data['streamer_z_min'][0],
        op='<')
    return idx


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       CALCULATE DATA                                                #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calc_average_speed(data, k0=0, k1=1):
    ''' Return the average speed of the streamer between z=k0 d, and z=k1 d.
    '''
    if ('streamer_z_min' not in data) or ('sim_time' not in data):
        logger.debug(f'Cannot calculate average speed')
        return None

    z = np.array(data['streamer_z_min']).reshape(-1)
    t = np.array(data['sim_time']).reshape(-1)

    d = z.max()
    mask = (d * k0 <= z) * (z <= d * k1)

    if mask.sum() > 0:
        dz = z[mask].max() - z[mask].min()
        dt = t[mask].max() - t[mask].min()
        return dz / (dt + eps)
    else:
        return None


def calc_gap_fraction(data):
    ''' Return the fraction of the gap propagated for each iteration.
    '''
    if ('streamer_z_min' not in data) or ('gap_size' not in data):
        logger.debug(f'Cannot calculate gap fraction')
        return None
    z = np.array(data['streamer_z_min']).reshape(-1)
    d_gap = get_sim_input(data)['gap_size']
    return (d_gap - z) / d_gap


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#                                                                     #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def inception_occurred(data):
    ''' Return True if inception has occurred. '''
    crit_avalanche = find_first_avalanche(data)
    if crit_avalanche is None:
        logger.debug('Cannot calculate inception.')
        return None
    else:
        return crit_avalanche > -1


def breakdown_occurred(data):
    ''' Return True if breakdown has occurred. '''
    if 'streamer_z_min' not in data:
        logger.debug('Cannot calculate breakdown.')
        return None
    z = np.array(data['streamer_z_min']).reshape(-1)
    # Casted explicitly to bool due to issue with json
    return bool(z.min() < z.max()*0.05)

#
