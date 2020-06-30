#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Plotting a matrix gives a good visual representation of differences.
    This is used to illustrate differences in:
    - simulation parameters
    - simulation termination
'''

# General imports
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import logging
from collections import OrderedDict
from pathlib import Path

# Import from project files
from .. import tools
from .. import sim_parameters
from ..load_data import LoadedData
from . import combination_tools

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

paramr = sim_parameters.paramr


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#   MATRIX PLOTTER                                                    #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class MatrixPlotter():

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Static methods                                            #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # list of options to be evaluated
    # using classmethod since class static variables are not inherited
    @classmethod
    def _get_opt_bool(cls):
        # these keywords may also be used for float or string
        # setting a kw to true here should then use the defaults
        opts = dict()
        opts['title']       = 'Show default title'
        opts['xlabel']      = 'Show x-axis label'
        opts['zlabel']      = 'Show z-axis label'
        opts['figure']      = 'Save the figure'
        opts['triml']       = 'Trim equal part of xlabels from left'
        opts['trimr']       = 'Trim equal part of xlabels from right'
        opts['save_data']   = 'Save plotted data as json-file'
        return opts

    @classmethod
    def _get_opt_float(cls):
        opts = dict()
        return opts

    @classmethod
    def _get_opt_string(cls):
        opts = dict()
        opts['fpath']      = 'Set output file path'
        opts['fpath_head'] = 'Set output file head'
        opts['folder']     = 'Set output file folder'
        opts['title']      = 'Set plot title'
        opts['xlabel']     = 'Set x-axis label'
        opts['zlabel']     = 'Set z-axis label'
        opts['reverse']    = 'Change x/y-axes'
        return opts

    @classmethod
    def opt2kwargs(cls, options=None):
        return tools.opt2kwargs(
            options=options,
            bools=cls._get_opt_bool(),
            floats=cls._get_opt_float(),
            strings=cls._get_opt_string(),
            )

    @classmethod
    def opt2help(cls):
        return tools.opt2help(
            bools=cls._get_opt_bool(),
            floats=cls._get_opt_float(),
            strings=cls._get_opt_string(),
            )

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Initialize                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def __init__(self,
                 # control output
                 fpath=None, folder=None, fpath_head=None,
                 # control axis
                 title=None, xlabel=None, zlabel=None,
                 reverse=False,
                 figure=True, save_data=False,
                 # control data
                 triml=False, trimr=False
                 ):
        logger.debug('Initiating ' + str(self.__class__.__name__))

        # get input / set defaults / sub-class to override if needed
        self.title = title
        self.fpath = fpath
        self.fpath_head = fpath_head
        self.folder = folder
        self.xlabel = xlabel
        self.zlabel = zlabel
        self.reverse = reverse

        self.triml = triml
        self.trimr = trimr

        # save figure and/or data
        self.figure = figure
        self.save_data = save_data

        # create figure
        self.fig = None
        self.axes = None        # the axes to be used

        # dict of data loaded from each file
        self.loaded_data = {}

        # plot data
        self.pd = {}

    def _clear_fig(self):
        '''Delete any axis and clear figure. '''
        if self.axes is not None:
            for ax in self.axes:
                self.fig.delaxes(ax)
        if self.fig is not None:
            self.fig.clf()

    def _set_fig(self):
        '''Create figure and add axes.'''
        self.fig = Figure()  # create new figure
        gs = GridSpec(1, 1, figure=self.fig)
        self.ax_xz = self.fig.add_subplot(gs[0])   # add axis
        self.axes = [self.ax_xz]                   # store axis list

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def add_fpath(self, fpath):
        ''' Load fpath, append data to be parsed later.
        Set fpath, if needed.

        Parameters
        ----------
        fpath : Path or str
                path to parse for data
        '''
        # set fpath by first given, if not set yet
        fpath = Path(fpath)
        self.set_fpath(fpath=fpath)

        # let subclass chose what to do with the data
        loaded_data = self.load_fpath(fpath)
        self.loaded_data[fpath] = loaded_data
        logger.debug(f'Loaded: {fpath}')

    def set_fpath(self, fpath=None):
        ''' Set fpath. '''
        if self.fpath is None:
            self.fpath = tools.get_fpath(
                {'fpath': fpath}, head=self.fpath_head, folder=self.folder)
            logger.log(5, f'Set fpath: {self.fpath}')

    def trim(self, xy_tups, triml=None, trimr=None):
        ''' Trim equal part of xlabels from left and/or right. '''

        # compile a set of all labels, get indexes for equal start and end
        xi_set = set(xi for (xi, yi) in xy_tups)
        lidx = _get_idx_equal_chars(xi_set, look_from='left')
        ridx = _get_idx_equal_chars(xi_set, look_from='right')

        # use self, if not given
        triml = self.triml if (triml is None) else triml
        trimr = self.trimr if (trimr is None) else trimr

        logger.log(5, f'Got triml {triml} and trimr {trimr}')

        # set to 0 if not True
        lidx = 0 if not triml else lidx
        ridx = 0 if not trimr else ridx

        logger.log(5, f'Got lidx {lidx} and ridx {ridx}')

        xy_tups_out = []
        for (xi, yi) in xy_tups:
            xi = str(xi)
            xi = xi[lidx:(len(xi)-ridx)]
            xy_tups_out.append((xi, yi))

        return xy_tups_out

    def load_fpath(self, fpath):
        ''' Load fpath and return loaded_data. '''
        # this method need to be defined in sub-class
        raise NotImplementedError('Plotter need to be sub-classed')

    def parse_data(self, data):
        '''Parse loaded_data and return xy_tups. '''
        # this method need to be defined in sub-class
        raise NotImplementedError('Plotter need to be sub-classed')

    def save_plotted_data(self):
        ''' Save plotted data to file. '''
        path = str(Path(self.fpath).with_suffix('.json'))
        tools.json_dump(self.pd, path)
        logger.info('Saved {}'.format(path))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage plotting methods (plot all data)                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def plot_all_pd(self):
        ''' Plot all data in `self.pd` to the axis.
        '''
        # parse data
        xy_tups = self.parse_data(self.loaded_data)
        xy_tups = self.trim(xy_tups)
        self.pd['xy_tups'] = xy_tups
        self.pd['files'] = list(self.loaded_data)

        # define xy-order
        if self.reverse:      # reverse xy
            xy_tups = [(y, x) for x, y in xy_tups]

        xlabs = OrderedDict()  # use ordered dict as ordered set
        ydict = OrderedDict()  # which x'es that are included in each y
        for x, y in xy_tups:   # initiate
            xlabs[x] = []
            ydict[y] = []
        for x, y in xy_tups:   # build
            ydict[y].append(x)

        ind = [i + 0.0 for i in range(len(ydict))]
        pad = 0.1       # padding between bar blocks
        left = pad      # start plotting from here

        # plot bar at each label containing each
        xlabs = sorted(xlabs)  # seems like sorting here is needed!
        for x in xlabs:
            # get true/false for each y-label
            widths = [(x in xl) for (y, xl) in ydict.items()]
            # change true/false to width of bar
            widths = [int(w) * (1 - 2 * pad) for w in widths]
            # plot widths for all indecies
            self.ax_xz.barh(ind, widths, left=left)
            # advance left side of all bars
            left += 1

        # correct limits and ticks
        # offset xtics by 0.5
        xind = [i + 0.5 for i in range(len(xlabs))]
        self.ax_xz.set_xticks(xind)
        self.ax_xz.set_xticklabels(xlabs, rotation=90)
        self.ax_xz.set_yticks(ind)
        self.ax_xz.set_yticklabels(ydict)
        self.ax_xz.set_xlim(left=0, right=(left - pad))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Finalize plot                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def set_labels(self):
        '''Control legend, title, and axis labels. '''
        logger.debug('Setting labels')
        if self.xlabel:
            self.ax_xz.set_xlabel(self.xlabel)
        if self.zlabel:
            # add to only one axis, 0 for furthest left
            self.axes[0].set_ylabel(self.zlabel)
        if self.title == 'fpath':
            self.title = str(Path(self.fpath).stem)
        if self.title:
            self.fig.suptitle(self.title)

    def save_plot(self, close=True):
        ''' Save added data as plotted figure, and/or json-file.
        Create figure, add plot, set labels, save figure.
        '''
        if self.fpath is None:
            logger.error('Error: cannot save plot without name.')
            return
        self._set_fig()                 # create the figure
        self.plot_all_pd()              # plot to figure
        self.set_labels()
        logger.info('Saving: {}'.format(self.fpath))
        if self.figure:
            Canvas(self.fig).print_figure(self.fpath)
        if close:
            self._clear_fig()
        if self.save_data:
            self.save_plotted_data()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   PARAMETER PLOTTER                                                 #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class ParameterPlotter(MatrixPlotter):
    '''Create a matrix plot of input parameters. '''

    @classmethod
    def _get_opt_string(cls):
        opts = super()._get_opt_string()
        opts['mode'] = 'Which keys to pick: `all`, [`varied`]'
        opts['keys'] = 'List of keys to plot, overrides mode'
        return opts

    @classmethod
    def _get_opt_bool(cls):
        opts = super()._get_opt_bool()
        opts['random_seed'] = 'Include random seed, `off`, [`on`]'
        return opts

    def __init__(self, keys=None, mode=None, random_seed=True, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'parameters'

        # self.title = 'Some title' if self.title is True else self.title
        # self.xlabel = xlabel
        # self.zlabel = zlabel

        self.random_seed = random_seed
        self.keys = None if keys is None else keys.split('_')
        self.mode = 'varied' if mode is None else mode

    def load_fpath(self, fpath):
        ''' Load input data from fpath and return as dict.

        Parameters
        ----------
        fpath : str or Path
            path to file to load from (json/log/pkl/input)

        Returns
        -------
        dict
            simulation input parameters
        '''
        return load_input_from_file(fpath)

    def parse_data(self, loaded_data):
        '''Parse loaded_data and return xy_tups.

        x is a filename.
        y is a parameter-value-label.

        Parameters
        ----------
        loaded_data : dict
            {fpath : simulation input parameters}

        Returns
        -------
        xy_tups : list(tuple(x, y), ...)
            list of tuples of labels
        '''
        if not loaded_data:
            logger.warning('Nothing to plot.')
            return []

        # get dict mapping keys and values to files
        # note: the structure is {parameter: {value: filename}}
        key_val_files = combination_tools.get_x_dict(
            loaded_data, mode=self.mode, x_keys=self.keys)

        # remove random seed from keys
        if not self.random_seed and 'random_seed' in key_val_files:
            key_val_files.pop('random_seed')

        # compile the list of xy tuples of labels to be plotted later
        xy_tups = []
        for k in key_val_files:
            logger.log(5, f'Parsing {k}: {key_val_files[k]}')
            if not key_val_files[k]:
                logger.info(f'No data for {k}')
            for v, files in sorted(key_val_files[k].items()):
                # set y label for each value and set of files
                # create options
                svs = paramr[k].get_val_str(val=v)  # symbol value str
                vs = paramr[k].get_val_str(val=v, symbol=False)
                lab = {}
                lab['lab_svs'] = f'{paramr[k].label}: {svs}'
                lab['svs'] = f'{svs}'
                lab['vs'] = f'{vs}'
                lab['lab_v'] = f'{paramr[k].label}: {v}'
                lab['lab_vs'] = f'{paramr[k].label}: {vs}'
                # idea: make option for user to set label
                # set label
                ylab = lab['svs']

                # get x label for each file in "value set"
                for file in files:
                    # set x lab here
                    file = str(file)
                    xlab = file     # keep entire filename
                    xy_tups.append((xlab, ylab))

        # return the tuples
        logger.log(5, f'Compiled xy_tups: {xy_tups}')
        return xy_tups


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#   STOP REASON PLOTTER                                               #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class StopReasonPlotter(MatrixPlotter):
    '''Create a matrix plot of stop reasons. '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fpath_head = 'stop'

        # self.title = 'Some title' if self.title is True else self.title
        # self.xlabel = xlabel
        # self.zlabel = zlabel

    def load_fpath(self, fpath):
        ''' Load log data from fpath and return stop reason.

        Parameters
        ----------
        fpath : str or Path
            path to file to load from (.log)

        Returns
        -------
        reason : str
            simulation stop reason
        '''
        return get_stop_reason(fpath)


    def parse_data(self, stop_reasons):
        ''' Parse loaded_data and return xy_tups.

        x is a filename.
        y is a stop reason.

        Parameters
        ----------
        stop_reasons :   dict
                         {fpath: reason}

        Returns
        -------
        xy_tups : list(tuple(x, y), ...)
            list of tuples of labels
        '''
        if not stop_reasons:
            logger.warning('Nothing to plot.')
            return []

        xy_tups = [(k, v) for k, v in stop_reasons.items()]

        # return the tuples
        logger.log(5, f'Compiled xy_tups: {xy_tups}')
        return xy_tups


# store a dict of available plotters
plotters = {
    'parameter': ParameterPlotter,
    'stopreason': StopReasonPlotter,
}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#   MANAGE PARAMETERS (simulation input)                              #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def load_input_from_file(fpath):
    ''' Load input data from fpath and return as dict.

    Parameters
    ----------
    fpath : str or Path
        path to file to load from (json/log/pkl/input)

    Returns
    -------
    dict
        simulation input parameters
    '''
    fpath = Path(fpath)

    # both json and input are straight forward to load
    if fpath.suffix == '.json':
        loaded_input_data = tools.json_load(fpath)
    elif fpath.suffix == '.input':
        loaded_input_data = tools.json_load(fpath)

    # parse log file
    elif fpath.suffix == '.log':
        logger.log(5, f'Parsing: {fpath}')
        # lines will be parsed as json later
        lines = '{\n'
        with fpath.open('r') as f:
            # skip the header
            for l in f:
                if l.startswith('{'):
                    break
            # keep the part that is the input parameters
            for l in f:
                lines += l
                if l.startswith('}'):
                    # skip the rest
                    break
        # sneaky way of getting json
        loaded_input_data = tools.json_loads(lines)

    # load saved data and find simulation input in its header
    elif fpath.suffix == '.pkl':
        loaded_input_data = LoadedData(fpath, header_only=True).sim_input

    else:
        # just return an empty dict for wrong file types
        logger.warning(f'Warning. Cannot parse files of type {fpath.suffix}')
        loaded_input_data = OrderedDict()

    # test for correctly loaded data type
    if not isinstance(loaded_input_data, dict):
        logger.warning(f'Wrong data for format. Discarding: {fpath}')
        logger.debug(f'Wrong format type: {type(loaded_input_data)}')
        loaded_input_data = OrderedDict()

    # test for content
    usable_keys = [k for k in loaded_input_data if k in paramr]
    logger.log(5, f'Usable keys: {usable_keys}')
    if not usable_keys:
        logger.warning(f'No usable parameter keys. Discarding: {fpath}')
        loaded_input_data = OrderedDict()

    logger.log(5, f'Returning {loaded_input_data}')
    return loaded_input_data


def _get_idx_equal_chars(strlst, look_from='left'):
    ''' Compare strings and return how many they have in common.

    Parameters
    ----------
    strlst : lst(str)
        list of strings
    look_from :  str
        look from 'left' or 'right'

    Returns
    -------
    idx
        number of equal characters from left
    '''
    # force everything to string (and create a new list)
    if look_from == 'left':
        strlst = [str(s) for s in strlst]
    elif look_from == 'right':
        strlst = [reversed(str(s)) for s in strlst]
    else:
        logger.error(f'Cannot look from: {look_from}.')
        return -1

    # create a set of every n'th character in the strings
    # set of length one is true, rest false
    bools = list((len(set(t)) == 1) for t in zip(*strlst))
    bools.append(False)
    idx = bools.index(False)  # get index of first false
    return idx


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#   STOP REASONS                                                      #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_stop_reason(fpath):
    ''' Load log data from fpath and return stop reason.

    Parameters
    ----------
    fpath : str or Path
        path to file to load from (.log)

    Returns
    -------
    string
        simulation stop reason
    '''
    fpath = Path(fpath)
    reason = ''         # reason for stop
    with fpath.open('r') as f:
        for line in f:  # find the line above the 'reason'
            pos = line.find('SIMULATION ENDED')
            if pos != -1:   # if found
                reason = f.readline()[pos:]  # next line, final part
                pos = reason.find('(')
                # reason = (reason[:pos], reason[pos+1:-2])
                reason = reason[:pos]   # remove 'reason argument'
    reason = reason or 'UNKNOWN'        # if not found
    return reason


#
