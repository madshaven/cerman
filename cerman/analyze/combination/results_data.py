#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Parse several simulation save files and gather the results in a database.
'''

# General imports
import numpy as np
from pathlib import Path
import logging
import re

# Import from project files
from ..load_data import LoadedData
from .. import tools

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       COMBINED DATA                                                 #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ResultsData(object):
    ''' A class for containing parsed data from several simulations.

    The main parameters of this class are:
    fpath: the path to the data file
    spaths: dict {save_file.stem: Path}.
    inptdd: dict {save_file.stem: input parameters}
    statdd: dict {save_file.stem: parsed results data}

    The data file is loaded (or created) if a path is provided to class init.
    Otherwise, the path is inferred from the first save file added.
    '''

    @classmethod
    def _get_opt_bool(cls):
        opts = dict()
        return opts

    @classmethod
    def _get_opt_float(cls):
        opts = dict()
        return opts

    @classmethod
    def _get_opt_string(cls):
        opts = dict()
        opts['fpath'] = 'Set archive file path'
        opts['mode'] = '[`add`], `remove`, `reload`, `reload_all`'
        return opts

    @classmethod
    def opt2kwargs(cls, options=None):
        return tools.opt2kwargs(
            options=options,
            strings=cls._get_opt_string(),
            )

    @classmethod
    def opt2help(cls):
        return tools.opt2help(
            bools=cls._get_opt_bool(),
            floats=cls._get_opt_float(),
            strings=cls._get_opt_string(),
            )

    def __init__(self, fpath=None, mode=None):
        ''' Create or load a new database.

            Workflow:
            If `fpath`.suffix is `pkl` or `stat`, remove it.
            If `fpath` is None, infer filename from fist file added.
            If `fpath.stat` is is a file, try to load it.
            If `fpath.pkl` is a file, infer filename from it.
            If `fpath.stat` does not exist, create it.
        '''

        self._clean()
        self.fpath = None
        self._is_initiated = False
        if fpath is not None:
            self._init(fpath)
        self.mode = 'add' if mode is None else mode

    def _init(self, fpath):
        ''' Set fpath. Initiate file, if needed. Load file.
        '''
        self._is_initiated = True  # NB! Can be set to False again below!
        fpath = Path(fpath)
        if (fpath.suffix == '.pkl') or (fpath.suffix == '.stat'):
            logger.debug('Stripping suffix from given path.')
            fpath = fpath.with_suffix('')

        if fpath.with_suffix('.pkl').exists():
            fpath = fpath.with_suffix('.pkl')
            self._infer_fpath(fpath)  # changes self.fpath
            fpath = self.fpath.with_suffix('')  # remove .stat

        if fpath.with_suffix('.stat').exists():
            fpath = fpath.with_suffix('.stat')
            self._init_from_file(fpath)

        elif not fpath.with_suffix('.stat').exists():
            fpath = fpath.with_suffix('.stat')
            self._init_new_file(fpath)

        else:
            self._is_initiated = False
            logger.error(f'Error! Cannot initiate from {fpath}')

    def __getitem__(self, arg):
        # make the class behave like its dict loaded by json
        if arg == 'fpath':
            return self.fpath
        elif arg == 'spaths':
            return self.spaths
        elif arg == 'statdd':
            return self.statdd
        elif arg == 'inptdd':
            return self.inptdd
        else:
            raise KeyError(arg)

    def modify(self, fpath, mode=None):
        # let's keep the logic below simple, only one option is performed
        # only `add` should be true by default

        fpath = Path(fpath)
        if mode is None:
            mode = self.mode

        if not self._is_initiated:
            self._init(fpath)
        if not self._is_initiated:
            # note: _init above gives a warning if unsuccessful
            return

        # perform action
        if mode == 'add':
            if fpath.stem not in self.spaths:
                self._add_data(fpath)
            else:
                logger.info(f'Already added {fpath}')
                logger.debug('Use `reload` to force reloading.')
        elif mode == 'reload':
            self._add_data(fpath)
        elif mode == 'remove':
            self._remove_data(fpath)
        elif mode == 'reload_all':
            self._reload_all()
        else:
            logger.info(f'Nothing to do. `{mode}` is not a valid option.')

    def _clean(self, fpath=None):
        # initiate / reset variables
        self.spaths = dict()    # save paths
        self.statdd = dict()    # combined data
        self.inptdd = dict()    # input data

    def _infer_fpath(self, fpath):
        '''Parse the given fpath to set: fname, sid, fpath.

        Splits fpath.stem, assuming:
        stem + _###_ + saveid
        (longer serial numbers are also allowed)
        '''

        fpath = Path(fpath)
        pattern = r'(.+)_[0-9][0-9][0-9]+_(.+).pkl'
        m = re.match(pattern, str(fpath))

        fname = m.group(1) + '_' + m.group(2)
        self.fpath = fpath.parent / (fname + '.stat')

        logger.log(5, f'Got           : {fpath}')
        logger.debug(f'Found fpath   : {self.fpath}')

    def _init_new_file(self, fpath):
        '''Initiate the class by creating a new file. '''

        fpath = Path(fpath)
        if fpath.suffix != '.stat':
            fpath = fpath.with_suffix('.stat')

        logger.info('Initiating new file {}'.format(str(fpath)))
        try:
            fpath.touch(exist_ok=False)
            self.fpath = fpath
            self._clean()               # initiating dicts
            self.dump()

        except BaseException as ex:
            logger.info('Could not create file {}'.format(str(fpath)))

            msg = 'An exception of type {0} occurred. Arguments:\n{1!r}'
            logger.info(msg.format(type(ex).__name__, ex.args))
            logger.log(5, 'Stack trace:', exc_info=True)

    def _init_from_file(self, fpath):
        '''Initiate the class by loading data from a file. '''

        logger.info('Initiating from {}'.format(fpath))
        try:
            data = tools.json_load(fpath)
            self.fpath = Path(data['fpath'])
            self.spaths = dict(
                (k, Path(v)) for k, v in data['spaths'].items())
            self.statdd = data['statdd']
            self.inptdd = data['inptdd']

        except BaseException as ex:
            logger.info('Loading data failed')

            msg = 'An exception of type {0} occurred. Arguments:\n{1!r}'
            logger.info(msg.format(type(ex).__name__, ex.args))
            logger.log(5, 'Stack trace:', exc_info=True)

    def _add_data(self, fpath):
        # load and add data from fpath

        fpath = Path(fpath)
        logger.info('Adding data {}'.format(fpath))
        try:
            loaded_data = LoadedData(fpath)
            statdd = parse_data(loaded_data)
            inptdd = loaded_data.sim_input
            loaded_data = {}    # clear to free memory

        except BaseException as ex:
            logger.info('Failed loading {}'.format(fpath))

            ex_msg = 'An exception of type {0} occurred.\nArguments: {1!r}'
            logger.debug(ex_msg.format(type(ex).__name__, ex.args))
            logger.log(5, 'Stack trace:', exc_info=True)
            return

        # if try succeeded
        key = fpath.stem
        self.spaths[key] = fpath
        self.statdd[key] = statdd
        self.inptdd[key] = inptdd

        # pretty print of dict, for low level logging
        if logger.getEffectiveLevel() <= 5:
            tools.print_dict(self.statdd[fpath.stem])

        self.dump()  # save

    def _remove_data(self, fpath):
        # remove data from fpath
        logger.info('Removing data {}'.format(fpath))

        # if try succeeded
        if fpath.stem in self.spaths:
            self.spaths.pop(fpath.stem, None)
            self.statdd.pop(fpath.stem, None)
            self.inptdd.pop(fpath.stem, None)
            self.dump()  # save
        else:
            msg = 'Cannot remove a file that is not in the database: {}'
            logger.info(msg.format(fpath))

    def dump(self):
        # Dump all class data to file.
        data = {}
        data['fpath'] = str(self.fpath)
        data['spaths'] = dict((k, str(v)) for k, v in self.spaths.items())
        data['statdd'] = self.statdd
        data['inptdd'] = self.inptdd

        logger.debug('Dumping data to {}'.format(self.fpath))
        # pretty print of dict, for low level logging
        if logger.getEffectiveLevel() <= 5:
            tools.print_dict(data)
        tools.json_dump(data, str(self.fpath))

    def _reload_all(self):
        # get all possible paths
        logger.debug('Reloading all available data')
        for key, fpath in sorted(self.spaths.items()):
            self._add_data(fpath)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       PARSE LOADED DATA                                             #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def parse_data(loaded_data):
    ''' Parse data loaded from a save file. Return results as a dict.
        See 'results_repr' for how these are represented.

    Parameters
    ----------
    loaded_data : LoadedData
        data from a save file

    Returns
    d : dict
        {results_key: value}
    '''
    # todo: better handle error(s) if data cannot be found
    # idea: tie this function closer together with 'results_repr'
    # idea: add averages, deviations, min, max, as appropriate for:
    #       - gap percent
    #       - Q, dQ
    #       - avalanche time
    #       - simulation loop time (step time)
    #       - epot
    #       - estr
    #       - headsno
    #       - seedsno
    #       - avalancheno
    #       - crit seeds
    #       - headsscale
    #       - new head time

    logger.debug('Parsing data.')
    data = loaded_data.data

    # todo: improve error handling
    if int(data.get('no', [-1])[-1]) == 0:
        raise IndexError('Only initiation iteration available.')

    # results data
    d = {}

    logger.log(5, 'Parsing data - meta')
    d['no'] = int(data.get('no', [-1])[-1])
    d['sim_time'] = float(data.get('sim_time', [-1])[-1])
    d['cpu_time'] = float(data.get('cpu_time', [-1])[-1])
    d['streamer_z_min'] = float(data.get('streamer_z_min', [-1])[-1])

    def _add_key_idx_no_time(key, idx):
        d[f'{key}_no'] = data['no'][idx]
        d[f'{key}_idx'] = idx
        d[f'{key}_time'] = data['sim_time'][idx]

    # Notes:
    #   Initiation time and distance can be calculated exact
    #   given the initial seed distribution.
    #   To be compared with the actual result? (validate dt)
    # first avalanche
    logger.log(5, 'Parsing data - first avalanche')
    idx = loaded_data.find_first_avalanche()
    _add_key_idx_no_time(key='first_avalanche', idx=idx)

    # fist added avalanche
    logger.log(5, 'Parsing data - first added')
    idx = loaded_data.find_first_added_avalanche()
    _add_key_idx_no_time(key='first_added_avalanche', idx=idx)

    # first z-change
    logger.log(5, 'Parsing data - first z-change')
    idx = loaded_data.find_first_z_change()
    _add_key_idx_no_time(key='first_z_change', idx=idx)
    d['first_z_change_val'] = (
        data['streamer_z_min'][0] -
        data['streamer_z_min'][d['first_z_change_idx']])

    # dz
    logger.log(5, 'Parsing data - dz')
    dz = np.array(data.get('streamer_z_min'))
    dz = dz[:-1] - dz[1:]
    any_jumps = (dz > 0).sum() > 0
    d['dz_average'] = 0 if not any_jumps else np.average(dz[dz > 0])
    d['dz_std'] = 0 if not any_jumps else  np.std(dz[dz > 0])
    d['dz_sum'] = dz.sum()
    d['dz_min'] = dz.min()
    d['dz_max'] = dz.max()
    gf = loaded_data.calc_gap_fraction()
    d['gap_fraction'] = gf if gf is None else gf.max()

    # Bool
    logger.log(5, 'Parsing data - inception/breakdown')
    d['inception_occurred'] = loaded_data.inception_occurred()
    d['breakdown_occurred'] = loaded_data.breakdown_occurred()
    # idea: add bools for modes

    # Speed
    logger.log(5, 'Parsing data - speed')
    cas = loaded_data.calc_average_speed
    d['speed'] = cas(k0=0.00, k1=1.00)
    d['speed_Q1'] = cas(k0=0.75, k1=1.00)
    d['speed_Q2'] = cas(k0=0.50, k1=0.75)
    d['speed_Q3'] = cas(k0=0.25, k1=0.50)
    d['speed_Q4'] = cas(k0=0.00, k1=0.25)
    d['speed_Q2Q3'] = cas(k0=0.25, k1=0.75)
    d['speed_Q1Q2'] = cas(k0=0.50, k1=1.00)
    d['speed_Q3Q4'] = cas(k0=0.00, k1=0.50)

    logger.debug('Returning parsed data.')
    return d

#
