#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Provide functionality for plotting results of several simulations.
'''

# General imports
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import logging
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path

# Import from project files
from .. import plotterXZ
from . import combination_tools
from . import results_repr
from .. import sim_parameters
from .. import tools
from ..cerman_rc_params import cerman_rc_params  # set plot style

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

paramr = sim_parameters.paramr
x_map = sim_parameters.map_keys
x_dep = sim_parameters.dependencies
resultr = results_repr.resultr
y_map = results_repr.map_keys


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE WRAPPER FOR RESULTS PLOTTER                            #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ResultsPlotterWrapper():
    ''' Design the k-dict, and send each y-key to the ResultsPlotter.

    The k_modes defines how a k_dict is created,
    and these depend on the keywords color, marker, and include.

    A number of k_dicts are created by permutation of
    k_keys (varied parameters) and k_modes (color, marker, include).

    The k_dicts are sorted according to having:
    mode than one color and more than one marker;
    more than one color;
    more than one marker;
    more than one data point;
    at least one data point.
    The first is chosen, and the rest are discarded.

    Change color, marker, or include to change the k_dict creation.
    '''

    @classmethod
    def _get_opt_bool(cls):
        opts = ResultsPlotter._get_opt_bool()
        opts['color'] = 'Differentiate with colors (default).'
        opts['marker'] = 'Differentiate with markers (default).'
        opts['include'] = 'Differentiate with includes/excludes (default).'
        return opts

    @classmethod
    def _get_opt_float(cls):
        opts = ResultsPlotter._get_opt_float()
        return opts

    @classmethod
    def _get_opt_string(cls):
        opts = ResultsPlotter._get_opt_string()
        opts['ykeys'] = 'Which result to plot (see results_info)'
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
    #       INITIATE                                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def __init__(self, x_keys, color=True, marker=True, include=True, **kwargs):

        self.km_color = bool(color)
        self.km_marker = bool(marker)
        self.km_include = bool(include)
        if not (self.km_color or self.km_marker):
            msg = 'Cannot plot without colors or markers. Using colors.'
            logger.warning(msg)
            self.km_color = True

        y_keys = kwargs.get('ykeys', '').split('_')
        self._set_x_keys(x_keys)      # x-axis parameter
        self._set_y_keys(y_keys)      # y-axis result
        self._set_k_modes()           # color/marker/include/exclude
        self.kwargs = kwargs          # kwargs for ResultsPlotter

    def add_data(self, results_data):
        # find/choose/set: legend parameters
        self._set_k_keys(results_data)
        self.results_data = results_data

    def save_plot(self):
        # permutate of x-keys and y-keys
        self._compile_plot_kwarg_list(self.kwargs)
        for plot_func in self.plot_func_gen():
            plot_func()

    def _set_x_keys(self, x_keys=None):
        # compare the given x-keys with the x-map, and use the valid ones

        if x_keys is None:
            x_keys = []
        if type(x_keys) is str:
            x_keys = x_keys.split('_')
        if type(x_keys) is not list:
            x_keys = [x_keys]

        # map abbreviations to correct keys
        invalid_keys = [x_key for x_key in x_keys if x_key not in x_map]
        x_keys = [x_map[x_key] for x_key in x_keys if x_key in x_map]
        if invalid_keys:
            msg = f'Ignoring invalid x-keys: {", ".join(invalid_keys)}'
            logger.warning(msg)

        if x_keys == []:
            x_keys = [x_map['needle_voltage']]
            msg = 'No valid x-keys. Setting x-key to `needle_voltage`.'
            logger.warning(msg)

        self.x_keys = x_keys

    def _set_y_keys(self, y_keys):
        # compare the given y-keys with the y-map, and use the valid ones

        if type(y_keys) is not list:
            y_keys = [y_keys]

        if ('help' in y_keys):
            self._print_valid_y_keys()
            y_keys.remove('help')

        if ('all' in y_keys):
            y_keys = list(y_map)

        # map abbreviations to correct keys
        invalid_keys = [y_key for y_key in y_keys if y_key not in y_map]
        y_keys = [y_map[y_key] for y_key in y_keys if y_key in y_map]

        if invalid_keys:
            msg = f'Ignoring invalid y-keys: {", ".join(invalid_keys)}'
            logger.warning(msg)

        if (y_keys == []):
            logger.warning('No valid y-keys.')
            self._print_valid_y_keys()

        self.y_keys = y_keys

    def _print_valid_y_keys(self):
        longest = max(len(k) for k in resultr)
        msg = '{k:>{l}} - {a:4} - {v}'
        info_string = '\n'.join(msg.format(
            k=k, a=resultr[k].abbr, v=resultr[k].label, l=longest)
            for k in resultr)

        logger.info('Available kinds are:')
        logger.info(msg.format(k='Key', a='Abbr', v='Name', l=longest))
        logger.info(msg.format(k='---', a='----', v='----', l=longest))
        logger.info(info_string)

    def _set_k_keys(self, results_data, k_keys_opt=None):
        # find the varied input parameters from compiled simulation data

        if k_keys_opt is None:
            k_keys_opt = []
        if type(k_keys_opt) is not list:
            k_keys_opt = [k_keys_opt]

        # get all varied input
        func = combination_tools.get_x_dict
        k_keys = list(func(results_data['inptdd'], mode='varied'))

        # remove random seed by default
        k_keys = [k for k in k_keys if k not in ['random_seed']]

        # add k_keys given as options
        k_keys += [x_map[k] for k in k_keys_opt]

        self.k_keys = k_keys

    def _set_k_modes(self):
        ''' k-modes defines how to color/mark/include/exclude parameters.
            k-modes are used for creating a k-dict.
        '''
        k_modes = []
        if self.km_color:
            k_modes += ['color']
        if self.km_marker:
            k_modes += ['marker']
        if self.km_include:
            k_modes += ['include']

        logger.log(5, 'k-modes {}'.format(k_modes))
        self.k_modes = k_modes

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       CREATE K-DICT                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _compile_plot_kwarg_list(self, plot_kwargs):
        # permutate x-keys and y-keys

        plot_tup_list = [
            (x_key, y_key, self.k_keys)
            for x_key in self.x_keys
            for y_key in self.y_keys
            ]

        logger.info('Created {} configuration(s)'.format(len(plot_tup_list)))
        logger.info('X-axis key(s): ' + ', '.join(self.x_keys))
        logger.info('Y-axis key(s): ' + ', '.join(self.y_keys))
        logger.info('Legend key(s): ' + ', '.join(self.k_keys))

        added_no = 0
        ignore_no = 0
        plot_kwargs_list = []
        for (x_key, y_key, k_keys) in plot_tup_list:
            # make a copy (see why below)
            plot_kwargs_ = dict(plot_kwargs, x_key=x_key, y_key=y_key)
            logger.info(f'Creating k-dict for: `{x_key}` and `{y_key}`')

            # set correct k_keys
            k_keys = [k_key for k_key in k_keys if k_key != x_key]
            # enable plotting, even when a k-dict is not needed
            if not k_keys:
                if x_key != 'gap_size':
                    k_keys = ['gap_size']
                else:
                    k_keys = ['needle_voltage']
                logger.info(f'No varied k-keys. Choosing: {k_keys[0]}')
            plot_kwargs_['k_keys'] = k_keys

            # k-dict check mode priority order
            k_cm_pri = {
                'multi_mc':     'at least two markers and two colors',
                'multi_color':  'at least two colors',
                'multi_marker': 'at least two markers',
                'multi_data':   'at least two values',
                'plotable':     'at least one value',
                'unplotable':   'no values',
                'all':          'anything',
                }
            # check mode dict k-dict list
            cmd_kdl = {k: [] for k in k_cm_pri}

            # create a generator for all possible k_dicts
            k_dict_gen = combination_tools.k_dict_gen(
                k_keys, inptdd=self.results_data['inptdd'], modes=self.k_modes)

            # create/append plot_kwargs for usable k-dicts
            for k_dict in k_dict_gen:
                out = combination_tools.check_k_dict(
                        k_dict, inptdd=self.results_data['inptdd'])
                for k, b in out.items():
                    cmd_kdl[k] += [k_dict] if b else []

            # log result of sorting
            for k in cmd_kdl:
                logger.debug(f'k-dicts `{k}` are {len(cmd_kdl[k])}')

            # Note! No k_dicts to append when this is chosen!
            cmd_kdl['unplotable'] = []

            # choose most stringent check mode and append those k-dicts
            for k in k_cm_pri:
                if len(cmd_kdl[k]) > 0:
                    k_check_mode = k
                    break
            for k_dict in cmd_kdl[k_check_mode]:
                plot_kwargs_['k_dict'] = k_dict
                plot_kwargs_list.append(plot_kwargs_.copy())

            # log reults
            _lk = len(cmd_kdl[k_check_mode])
            _la = len(cmd_kdl['all'])
            msg = f'Kept {_lk} and ignored {_la - _lk} plot configuration(s). '
            msg += f'Using {k_cm_pri[k_check_mode]}.'
            logger.info(msg)
        self.plot_kwargs_list = plot_kwargs_list

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       YIELD PLOT FUNCTIONS                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def plot_func_gen(self):
        # loop over the argument list and create plots
        for plot_kwargs in self.plot_kwargs_list:
            x_key = plot_kwargs.pop('x_key')
            y_key = plot_kwargs.pop('y_key')
            k_keys = plot_kwargs.pop('k_keys')
            k_dict = plot_kwargs.pop('k_dict')
            def plot_func():
                plotter = ResultsPlotter(**plot_kwargs)
                plotter.add_data(self.results_data, x_key, y_key, k_dict)
                plotter.save_plot()
            msg = f'Plotting {x_key} vs {y_key} for {", ".join(k_keys)}'
            plot_func.__doc__ = msg
            yield plot_func


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE RESULTS PLOTTER                                        #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ResultsPlotter(object):
    ''' Plot a combination of simulation results.

    Use one simulation parameter on the x-axis (given by x-key)
    and plot one result on the y-axis (given by y-key).

    If more than one simulation parameter is varied,
    then the legend will indicate their values (given by k-dict).

    Colors and markers may be used to show several varied parameters,
    or more results, on the same plot.

    For usable x-keys, see `sim_parameters`.
    For usable y-keys, see `results_repr`.
    A k-dict looks like {k_key: {k_val: mode}
    with `mode` in ['color', 'marker', 'include', 'exclude'].
    '''

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
        opts['scatter']     = 'Show scatter plot'
        opts['interpolate'] = 'Show interpolation'
        opts['statistics']  = 'Show average, plus/minus a deviation'
        opts['colors']      = 'Cycle colors (default)'
        opts['markers']     = 'Cycle markers'
        opts['logx']        = 'Show x-axis logarithmic'
        opts['logz']        = 'Show z-axis logarithmic'
        opts['title']       = 'Show default title'
        opts['figure']      = 'Save the figure'
        opts['save_data']   = 'Save plotted data as json-file'
        opts['legend']      = 'Show legend'
        opts['xlabel']      = 'Show x-axis label'
        opts['zlabel']      = 'Show z-axis label'
        return opts

    @classmethod
    def _get_opt_float(cls):
        opts = dict()
        opts['zmin']   = 'Set z-axis minimum'
        opts['zmax']   = 'Set z-axis maximum'
        opts['xmin']   = 'Set x-axis minimum'
        opts['xmax']   = 'Set x-axis maximum'
        return opts

    @classmethod
    def _get_opt_string(cls):
        opts = dict()
        opts['fpath']      = 'Set output file path'
        opts['folder']     = 'Set output file folder'
        opts['title']      = 'Set plot title'
        opts['legend']     = 'Set legend: [`full`], `reduced`, `off`'
        opts['xlabel']     = 'Set x-axis label'
        opts['zlabel']     = 'Set z-axis label'
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
            fpath=None, folder=None,
            # control axis
            title=None, legend=None,
            xlabel=True, xmin=None, xmax=None,
            zlabel=True, zmin=None, zmax=None,
            logx=False, logz=False,
            # plot type
            scatter=True, statistics=False, interpolate=True,
            # control plot
            figure=True, save_data=False,
            colors=True, markers=False,
            # other
            annotate=False,
            ### add below?
            # results_data, x_key, y_keys, k_dict
            **kwargs):
        logger.debug('Initiating ' + str(self.__class__.__name__))

        # get input / set defaults / sub-class to override if needed
        self.title = title  # IMa: true title can be y_key, set later -- resultr[y_key].label
        self.folder = folder
        self.fpath = fpath
        self.xlabel = xlabel
        self.zlabel = zlabel
        self.figure = figure
        self.save_data = save_data

        self.logx_on = bool(logx)
        self.logz_on = bool(logz)
        self.scatter_on = scatter
        self.statistics = bool(statistics)
        interpolate = ['a'] if interpolate is None else interpolate
        interpolate = ['a'] if interpolate is True else interpolate
        self.interpolate = [] if interpolate is False else interpolate
        if legend not in [None, 'full', 'reduced', 'off', True, False]:
            legend = None  # default
            logger.warning('Changed legend mode to default.')
        if legend in [None, 'full', True]:
            self.legend = 'full'
        else:
            self.legend = legend  # False or reduced

        # plot data
        # add one dict for each plotted data
        self.plot_data = []

        # plot limits, set by initiation
        # leave as 'none' to use plotted data instead
        self.zmin = zmin
        self.zmax = zmax
        self.xmin = xmin
        self.xmax = xmax

        # create figure
        self.fig = None
        self.axes = None        # the axes to be used

    def reset_mc_cc(self):
        ''' Reset the marker and color cycle iterators. '''
        self.mc = cerman_rc_params.marker_cycler()()  # invoke the iterator
        self.cc = cerman_rc_params.color_cycler()()

    def _clear_fig(self):
        '''Delete any axis and clear figure. '''
        if self.axes is not None:
            for ax in self.axes:
                self.fig.delaxes(ax)
        if self.fig is not None:
            self.fig.clf()

    def _set_fig(self):
        '''Create figure and add axes.'''
        self.reset_mc_cc()      # reset colors
        self._clear_fig()       # clear any present figure

        self.fig = Figure()  # create new figure
        gs = GridSpec(1, 1, figure=self.fig)
        self.ax_xz = self.fig.add_subplot(gs[0])   # add axis
        self.axes = [self.ax_xz]                   # store axis list


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def add_data(self, results_data, x_key, y_key, k_dict):
        ''' Parse data, append data to plot.
        Set fpath, llabel, and update origin, as needed.

        Parameters
        ----------
        data :  dict
                loaded simulation data
        '''

        self.results_data = results_data
        self.x_key = x_key
        self.y_key = y_key
        self.k_dict = k_dict

        if not self._validate_data(x_key, y_key, k_dict):
            return
        self.set_fpath(self.fpath)
        self.sort_results_data(results_data)
        self.sort_files_to_plot(k_dict)
        self.sort_colors_markers(k_dict)
        self.build_plot_data()

    def _validate_data(self, x_key, y_key, k_dict):
        is_valid = True
        if x_key not in paramr:
            logger.error(f'Error! x_key `{x_key}` not in `paramr`')
            is_valid = False
        if y_key not in resultr:
            logger.error(f'Error! y_key `{y_key}` not in `resultr`')
            is_valid = False
        for k_key in k_dict:
            if k_key not in paramr:
                logger.error(f'Error! k_key `{k_key}` not in `paramr`')
                is_valid = False
        if x_key in k_dict:
            k_dict.pop(x_key)
            logger.log(5, 'Removed x-key from k-dict.')
        return is_valid

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       GET FILENAME                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def set_fpath(self, fpath=None):
        '''Set fpath. Use fpath or set from data, literally.'''
        if fpath is not None:
            self.fpath = fpath
        else:
            tail = ''
            tail += '_logx' if (self.logx_on == True) else ''
            tail += '_logz' if (self.logz_on == True) else ''
            tail += '_' + paramr[self.x_key].abbr
            tail += '_' + resultr[self.y_key].abbr

            # add part of filename based on k-dict (color/marker/include)
            tail += '_'
            for k_key, k_item in self.k_dict.items():
                tail += paramr[k_key].abbr
                tail += '(c)' if (k_item[list(k_item)[0]] == 'color') else ''
                tail += '(m)' if (k_item[list(k_item)[0]] == 'marker') else ''
                nos = [str(i)
                       for i, k_val in enumerate(k_item)
                       if k_item[k_val] == 'include'
                       ]
                if nos:
                    tail += '({})'.format(','.join(nos))
                tail += '-'

            tail = tail[1:-1]      # remove first underscore and final dash
            logger.log(5, 'tail {}'.format(tail))
            self.fpath = tools.get_fpath(self.results_data, tail=tail)
        logger.log(5, f'Set fpath: {self.fpath}')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       SORT RESULTS DATA                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def sort_results_data(self, results_data):

        # get results_data
        self.statdd = results_data['statdd']   # stat results_data
        self.inptdd = results_data['inptdd']   # input results_data
        # get all d_keys, sorted by key and value
        self.x_key_val_dkey = combination_tools.get_x_dict(self.inptdd)

    def sort_files_to_plot(self, k_dict):
        # get sets of d_keys ['plot', 'color', 'marker', 'include', 'exclude']
        mode_dkeys = combination_tools.get_mode_dkeys(
            k_dict, self.inptdd, exclude=True)
        self.d_plot = mode_dkeys['plot']         # all keys to be plotted
        self.d_excl = mode_dkeys['exclude']      # all keys not to be plotted
        logger.log(5, f'Files to plot: {", ".join(self.d_plot)}')
        logger.log(5, f'Files to not plot: {", ".join(self.d_excl)}')

        if len(self.d_plot) == 0:
            logger.warning(f'No files to plot. {self.fpath}')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       SORT COLORS/MARKERS                                       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def sort_colors_markers(self, k_dict):
        self.reset_mc_cc()
        color = next(self.cc)['color']
        marker = next(self.mc)['marker']

        self.colored_tups = []   # (k_key, k_val)
        self.marked_tups = []    # (k_key, k_val)
        self.included_tups = []  # (k_key, k_val)
        self.color_dict = {}     # {k_key: {k_val: color}}
        self.marker_dict = {}    # {k_key: {k_val: marker}}

        for k_key in k_dict:
            # init dict for each key
            self.color_dict[k_key] = {}
            self.marker_dict[k_key] = {}

            for k_val in self.k_dict[k_key]:
                # set color/marker first, then update them
                self.color_dict[k_key][k_val] = color
                self.marker_dict[k_key][k_val] = marker

                if self.k_dict[k_key][k_val] == 'color':
                    self.colored_tups.append((k_key, k_val))
                    color = next(self.cc)['color']
                elif self.k_dict[k_key][k_val] == 'marker':
                    self.marked_tups.append((k_key, k_val))
                    marker = next(self.mc)['marker']
                elif self.k_dict[k_key][k_val] == 'include':
                    self.included_tups.append((k_key, k_val))
                elif self.k_dict[k_key][k_val] == 'exclude':
                    pass
                else:
                    logger.error('Implementation error, please recheck!')

        logger.log(5, f'color_dict {self.color_dict}')
        logger.log(5, f'marker_dict {self.marker_dict}')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       PLOT LEGEND                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def add_legend(self):
        logger.debug('Adding up legend')
        # plot legend for k-dict
        for k_key, k_item in self.k_dict.items():
            # plot label for each k_key
            modes = set(k_mode for (k_val, k_mode) in k_item.items())
            if ((self.legend == 'full') or
                    ('color' in modes) or
                    ('marker' in modes)
                    ):
                # add label for the k_key (input parameter)
                label = f'{paramr[k_key].label}'
                self.ax_xz.plot([], [], ls='', label=label)

            # add label for each k_val (parameter value)
            for k_val, k_mode in k_item.items():
                color = self.color_dict[k_key][k_val]
                marker = self.marker_dict[k_key][k_val]
                label = paramr[k_key].get_val_str(val=k_val, symbol=False)

                if k_mode == 'color':
                    self.ax_xz.plot([], [], color=color, label=label)
                elif k_mode == 'marker':
                    self.ax_xz.plot(
                        [], [], marker=marker, ls='', label=label, color='k')
                elif k_mode == 'include':
                    if self.legend == 'full':
                        self.ax_xz.plot(
                            [], [], ls='', marker=5, color='k', label=label)
                elif k_mode == 'exclude':
                    if self.legend == 'full':
                        self.ax_xz.plot(
                            [], [], ls='', marker='_', color='k', label=label)
                else:
                    logger.error(f'Error. Unknown k-mode {k_mode}')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       FUNCTIONS FOR BUILDING PLOT DATA                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _get_d_keys(self, mk_key, mk_val, ck_key, ck_val):

        # get keys to be marked or colored
        md_keys = set(self.x_key_val_dkey[mk_key][mk_val])
        cd_keys = set(self.x_key_val_dkey[ck_key][ck_val])
        # assume that other marks/colors need not be excluded
        # assume included is not needed, only exclusion
        d_keys = (md_keys & cd_keys) - self.d_excl
        return d_keys

    def _get_xy(self, d_keys):

        # Get, check, and scale data
        x = [paramr[self.x_key].get_val(self.inptdd[d_key]) for d_key in d_keys]
        y = [resultr[self.y_key].get_val(self.statdd[d_key]) for d_key in d_keys]

        # sort items to plot (exclude results that are `None`)
        xy = [(xi, yi) for (xi, yi) in zip(x, y) if yi is not None]
        xy = np.array(xy).reshape(-1, 2)
        xp = xy[:, 0]
        yp = xy[:, 1]

        # log and return output
        logger.log(5, 'files, {}'.format(d_keys))
        logger.log(5, 'x-key, {}'.format(self.x_key))
        logger.log(5, 'x-val, {}'.format(x))
        logger.log(5, 'xp   , {}'.format(xp))
        logger.log(5, 'y-key, {}'.format(self.y_key))
        logger.log(5, 'y-val, {}'.format(y))
        logger.log(5, 'yp   , {}'.format(yp))
        return xp, yp

    def _get_desc(self, mk_key, mk_val, ck_key, ck_val):

        desc = dict(
            marker=self.marker_dict[mk_key][mk_val],
            ls='',
            alpha=0.5,
            label=None,
            color=self.color_dict[ck_key][ck_val],
            )
        return desc

    def _xs_xtld(self, x=None, xtll=[]):
        if x is None:
            return xtll  # x-tick-label-list

        else:
            xtll += [xi for xi in x if xi not in xtll]  # add in place!
            xout = [xtll.index(xi) for xi in x]
            return xout

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       BUILD PLOT DATA                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def build_plot_data(self):
        # allow iteration if the list is empty
        marked_tups = self.marked_tups or [(None, None)]
        colored_tups = self.colored_tups or [(None, None)]

        logger.debug('Get and plot data')
        for mk_key, mk_val in marked_tups:
            for ck_key, ck_val in colored_tups:
                self._build_plot_data_work(mk_key, mk_val, ck_key, ck_val)

    def _build_plot_data_work(self, mk_key, mk_val, ck_key, ck_val):
        logger.log(5, 'mk_key, mk_val, ck_key, ck_val')
        logger.log(5, f'{mk_key}, {mk_val}, {ck_key}, {ck_val}')

        # use same keys for colors and marks, if one is missing
        if (mk_key, mk_val) == (None, None):
            mk_key, mk_val = ck_key, ck_val
        if (ck_key, ck_val) == (None, None):
            ck_key, ck_val = mk_key, mk_val
        if (mk_val is None) and (ck_val is None):
            msg = 'Marked or colored data is needed for plotting.'
            logger.error(msg)
            return

        # get names of usable results_data files (y-key is irrelevant)
        d_keys = self._get_d_keys(mk_key, mk_val, ck_key, ck_val)

        # get results_data to plot (x-key is global)
        x, y = self._get_xy(d_keys)

        # change x from string to index, if needed
        if x.size:
            if isinstance(x[0], str) or isinstance(x[0], np.str_):
                x = self._xs_xtld(x=x)
                y = y.astype(float)  # y get converted to

        # get color and marker
        desc = self._get_desc(mk_key, mk_val, ck_key, ck_val)

        # make a proper, long label for the data to be saved
        l_lab = ''
        if ck_key is not None:
            _l = paramr[ck_key].get_val_str(val=ck_val, symbol=True)
            l_lab += ', ' + _l
        if mk_key is not None:
            _l = paramr[mk_key].get_val_str(val=mk_val, symbol=True)
            l_lab += ', ' + _l
        l_lab = l_lab[2:]

        # append data
        pd = OrderedDict()
        # main data
        pd['title'] = self.title
        pd['l_lab'] = l_lab
        pd['x_lab'] = paramr[self.x_key].get_label()
        pd['y_lab'] = resultr[self.y_key].get_label()
        pd['x'] = str(np.array(x).tolist())
        pd['y'] = str(np.array(y).tolist())
        pd['desc'] = desc

        # other
        pd['y_key'] = self.y_key
        pd['x_key'] = self.x_key
        pd['mk_key'] = mk_key
        pd['mk_val'] = mk_val
        pd['ck_key'] = ck_key
        pd['ck_val'] = ck_val
        pd['d_keys'] = str(sorted(d_keys))  # json does not like sets

        # add
        self.plot_data.append(pd)
        logger.debug(f'Added {len(x)} data points for: {l_lab}')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage plotting methods (plot all data)                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def plot_all_pd(self):
        ''' Plot all data in `self.plot_data` to the axis.
        '''
        self.add_legend()
        for pd in self.plot_data:
            # data is saved as stings, so this looks a bit weird
            x = np.fromstring(pd['x'][1:-1], sep=', ')
            y = np.fromstring(pd['y'][1:-1], sep=', ')
            desc = pd['desc']
            self._plot(x, y, desc)

    def _plot(self, x, y, desc):

        if self.scatter_on:
            # add markers
            self.ax_xz.plot(x, y, **desc)

        # add interpolation
        if self.interpolate and np.unique(x).size > 1:
            logger.debug('Adding interpolation')
            combination_tools.ax_add_1d_interpolation(
                self.ax_xz, x, y, desc, kind='pchip', to_plot=self.interpolate,
                logx=self.logx_on, logy=self.logz_on)

        # add statistics
        if self.statistics:
            logger.debug('Adding statistics')
            combination_tools.ax_add_statistics(
                self.ax_xz, x, y, desc, to_plot=['a', 'p', 'm'])

        return desc['label']

    def save_plotted_data(self):
        ''' Save plotted data to file. '''
        fpath = str(Path(self.fpath).with_suffix('.json'))
        tools.json_dump(self.plot_data, fpath)
        logger.info(f'Saved {fpath}')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       FINALIZE PLOT                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def set_limits(self):
        ''' Set axes lin/log and set limits.
        '''
        logger.debug('Setting limits')

        if self.logx_on:
            self.ax_xz.set_xscale('log')
        if self.logz_on:
            self.ax_xz.set_yscale('log')
        else:
            self.ax_xz.set_ylim(bottom=0)

        # get matplotlib limits
        xmin = self.ax_xz.get_xlim()[0]
        xmax = self.ax_xz.get_xlim()[1]
        zmin = self.ax_xz.get_ylim()[0]
        zmax = self.ax_xz.get_ylim()[1]

        # override if user have defined values
        xmin = self.xmin if (self.xmin is not None) else xmin
        xmax = self.xmax if (self.xmax is not None) else xmax
        zmin = self.zmin if (self.zmin is not None) else zmin
        zmax = self.zmax if (self.zmax is not None) else zmax

        # set the limits
        self.ax_xz.set_xlim((xmin, xmax))
        self.ax_xz.set_ylim((zmin, zmax))

        xt, zt = self.axes[0].get_xlim(), self.axes[0].get_ylim(),
        logger.debug(f'Limits set: x=({xt}), z=({zt})')

    def set_labels(self):
        '''Control legend, title, and axis labels. '''
        logger.debug('Setting labels')
        if self.legend in ['full', 'reduced']:
            self.ax_xz.legend(numpoints=1, loc='best')

        if self._xs_xtld():  # returns empty list if x is not string values
            self.ax_xz.set_xticks(range(len(self._xs_xtld())))
            self.ax_xz.set_xticklabels(self._xs_xtld())

        if self.xlabel is True:
            self.xlabel = paramr[self.x_key].get_label()
        if self.xlabel:
            self.ax_xz.set_xlabel(self.xlabel)
        if self.zlabel is True:
            self.zlabel = resultr[self.y_key].get_label()
        if self.zlabel:
            self.ax_xz.set_ylabel(self.zlabel)

        if self.title == 'fpath':
            self.title = str(Path(self.fpath).stem)
        if self.title is True:
            self.title = resultr[self.y_key].label
        if self.title:
            self.fig.suptitle(self.title)

    def save_plot(self, close=True):
        ''' Save added data as plotted figure, and/or json-file.
        Create figure, add all plots, set limits, set labels, save figure.
        '''
        if self.fpath is None:
            logger.error('Error: cannot save plot without name.')
            return
        self._set_fig()                 # create the figure
        self.plot_all_pd()              # plot to figure
        self.set_limits()
        self.set_labels()
        logger.info('Saving: {}'.format(self.fpath))
        if self.figure:
            Canvas(self.fig).print_figure(self.fpath)
        if close:
            self._clear_fig()
        if self.save_data:
            self.save_plotted_data()


#
