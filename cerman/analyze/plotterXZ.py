#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Defines a superclass for plotting.

The class contains functions for most aspects of plotting.
Data extraction is performed by sub-class.

The goal is to automate creation of figures.
This approach enables figures to be kept while working,
and adding more data, similar to matplotlib.

This approach have been shifted towards extracting data first, then plot.
'''

# imports for plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import warnings

# imports for data manipulation
from statsmodels.nonparametric import smoothers_lowess
from scipy.interpolate import pchip_interpolate
import numpy as np
import logging
from pathlib import Path
from collections import OrderedDict

# import matplotlib.animation as mpla
# hack: mpla is imported when/if needed later in the code
#       it is used for movie creation
#       it stops when cerman is started in background

# Import from project files
from . import tools
from .cerman_rc_params import cerman_rc_params

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# todo: consider a cleaner base case class for plotting
#       let the base case be to read/import json-formatted data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE PLOTTER XZ                                             #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class PlotterXZ(object):
    '''Superclass for plotting.

    The workflow is like:
    plotter = PlotterXZ(**kwargs)
    plotter.add_data(loaded_data_1)  # load by other means
    plotter.add_data(loaded_data_2)  # parsing more data
    ...
    plotter.add_data(loaded_data_n)  # parsing more data
    plotter.save_plot()              # create the figure, plot, and save

    Inspect `../analyze_cm.py` for more details on use.
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
        opts['curve']       = 'Show curve plot'
        opts['scatter']     = 'Show scatter plot'
        opts['lowess']      = 'Show lowess plot'
        opts['interpolate'] = 'Show interpolation'
        opts['lowess_zx']      = 'Show lowess for x(z)'
        opts['interpolate_zx'] = 'Show interpolation for x(z)'
        opts['errorbar']    = 'Show errorbars'
        opts['minmaxbar']   = 'Show min-max bars'
        opts['minmaxfill']  = 'Show fill between min-max'
        opts['colors']      = 'Cycle colors (default)'
        opts['markers']     = 'Cycle markers'
        opts['logx']        = 'Show x-axis logarithmic'
        opts['logz']        = 'Show z-axis logarithmic'
        opts['title']       = 'Show default title'
        opts['xlabel']      = 'Show x-axis label'
        opts['zlabel']      = 'Show z-axis label'
        opts['figure']      = 'Save the figure'
        opts['movie']       = 'Create a movie'
        opts['frames']      = 'Save each frame of the movie'
        opts['tight']       = 'Use tight layout'
        opts['sparsify']    = 'Reduce the size of scatter plots'
        opts['save_data']   = 'Save plotted data as json-file'
        opts['legend']      = 'Show legend (default).'
        return opts

    @classmethod
    def _get_opt_float(cls):
        opts = dict()
        opts['xscale'] = 'Set scale of x-axis'
        opts['zscale'] = 'Set scale of z-axis'
        opts['zmin']   = 'Set z-axis minimum'
        opts['zmax']   = 'Set z-axis maximum'
        opts['xmin']   = 'Set x-axis minimum'
        opts['xmax']   = 'Set x-axis maximum'
        opts['lowess'] = 'Set lowess fraction'
        opts['fps']    = 'Set Frame Per Second (for movie)'
        opts['tscale'] = 'Set time scale (move time / simulation time)'
        opts['diffx'] = 'Set offset in x for each data series'
        opts['diffz'] = 'Set offset in z for each data series'
        opts['sparsify'] = 'Set number of bins in sparsify.'
        # todo: consider implementing:
        # opts['errorbar']   = 'Set errorbar deviations'
        # opts['minmaxbar']  = 'Set minmaxbar fraction'
        # opts['minmaxfill'] = 'Set minmaxfill alpha'
        return opts

    @classmethod
    def _get_opt_string(cls):
        opts = dict()
        opts['fpath']      = 'Set output file path'
        opts['fpath_head'] = 'Set output file head'
        opts['fpath_ext']  = 'Set output file extension'
        opts['folder']     = 'Set output file folder'
        opts['title']      = 'Set plot title'
        opts['legend']     = 'Legend `legend=v_rp_gs_unsorted_loc=best`'
        opts['annotate']   = 'Annotate plot, `annotate=x_y_text`.'
        opts['xlabel']     = 'Set x-axis label'
        opts['zlabel']     = 'Set z-axis label'
        opts['tlabel']     = 'Set time label'
        opts['fx']         = 'Add a line, fx=a_b_c --> a+bx+cx^2'
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
            fpath=None, fpath_head=None, fpath_ext=None, folder=None,
            # control axis
            title=None, legend=True,
            xlabel='X', xscale=1, xmin=None, xmax=None,
            zlabel='Z', zscale=1, zmin=None, zmax=None,
            diffx=0, diffz=0,
            logx=False, logz=False,
            # plot type
            curve=False, scatter=False,
            lowess=False, interpolate=False,
            lowess_zx=False, interpolate_zx=False,
            errorbar=False, minmaxbar=False, minmaxfill=False,
            # control movie
            movie=False, frames=False, fps=5,
            tscale=1e6, tlabel='\u00B5s',
            # control plot
            figure=True, save_data=False,
            colors=True, markers=False,
            tight=False, sparsify=None,
            # other
            fx=None, annotate=False,
            ):

        logger.debug('Initiating ' + str(self.__class__.__name__))

        # get input / set defaults / sub-class to override if needed
        self.title = title
        self.fpath_head = fpath_head
        self.fpath_ext = fpath_ext or mpl.rcParams['savefig.format']
        self.folder = folder
        self.fpath = fpath      # if None: set by add_data
        self.xlabel = xlabel
        self.zlabel = zlabel
        self.xscale = xscale
        self.zscale = zscale
        self.figure = figure
        self.save_data = save_data

        self.logx_on = bool(logx)
        self.logz_on = bool(logz)

        # parse legend commands
        self.legend_on = True
        self.legend_sorted = True
        self.legend_loc = 'best'    # location
        self.legend_fmt = None      # format of legend values
        self.legend_lkw = []        # label keywords (from sim parameters)
        if type(legend) is str:
            legend = legend.split('_')
            for key in legend:
                if key.startswith('loc='):
                    self.legend_loc = key[len('loc='):]
                elif key.startswith('fmt='):
                    self.legend_fmt = key[len('fmt='):]
                elif key == 'on':
                    self.legend_on = True
                elif key == 'off':
                    self.legend_on = False
                elif key == 'sorted':
                    self.legend_sorted = True
                elif key == 'unsorted':
                    self.legend_sorted = False
                else:
                    self.legend_lkw.append(key)
        if self.legend_lkw == []:
            self.legend_lkw = ['name']  # use filename as default

        # parse annotate commands
        # assume `annotate=x_y_text_format`
        self.annotate_on = False
        # self.annotate_format = 'some default'  # todo: implement this
        if type(annotate) is str:
            annotate = annotate.split('_')
            if len(annotate) < 3:
                msg = 'Give at least three arguments to `annotate`. Skipping.'
                logger.info(msg)
            else:
                self.annotate_on = True
                self.annotate_x = float(annotate[0])
                self.annotate_y = float(annotate[1])
                self.annotate_text = str(annotate[2])
                if len(annotate) > 3:
                    self.annotate_format = str(annotate[3])

        # define what to plot
        self.curve_on = curve
        self.scatter_on = scatter
        self.lowess_on = lowess
        self.lowess_frac = 0.2 if type(lowess) is not float else lowess
        self.interpolate_on = interpolate
        self.errorbar_on = errorbar
        self.minmaxbar_on = minmaxbar
        self.minmaxfill_on = minmaxfill
        self.sparsify_on = bool(sparsify)
        if (sparsify is None or sparsify is True):
            self.sparsify_no = 100
        else:
            self.sparsify_no = int(sparsify)

        self.lowess_zx_on = lowess_zx
        self.lowess_zx_frac = 0.2 if type(lowess_zx) is not float else lowess_zx
        self.interpolate_zx_on = interpolate_zx

        # define how to add a line
        # note, the line need to be added after other data to get limits
        self.fx = None
        if type(fx) is str:
            # fy(x) = sum([ai * x**i for i, ai in enumerate(self.fx)])
            self.fx = [float(ai) for ai in fx.split('_')]

        # plot limits, set by initiation
        # leave as 'none' to use plotted data instead
        self.zmin = zmin
        self.zmax = zmax
        self.xmin = xmin
        self.xmax = xmax

        # plot limits, set by plotted data
        # idea: why not just get these later instead?
        self._zmin = +1e10
        self._zmax = -1e10
        self._xmin = +1e10
        self._xmax = -1e10

        # possibly add offset to plots
        self.originx = 0
        self.originz = 0
        self.diffx = diffx
        self.diffz = diffz

        # plot setup
        # note: set label explicitly to None for all other than 'desc'
        #       or explicitly supress label when plotting
        self.desc         = {'lw': 2, 'alpha': 1}
        self.curve_desc   = {'ls': '-', 'marker': None, 'label': None}
        self.lowess_desc  = {'ls': '--', 'marker': None, 'label': None}
        self.interp_desc  = {'ls': '--', 'marker': None, 'label': None}
        self.scatter_desc = {'s': 5, 'marker': '.', 'label': None}
        self.xz_ratio = False  # set ratio only for equal plots
        self.symx = False  # symmetric xlim
        self.symz = False  # symmetric zlim
        self.colors = colors
        self.markers = markers

        # create figure
        self.fig = None
        self._fig_ghost = None  # for ghost axes
        self.axes = None        # the axes to be used
        self._axes_ghost = None    # for ghost axes
        self.tight = bool(tight)

        # plot data
        # append one list for each plot
        # idea: make a plot dict for all these instead (to_plot_data?)
        self.pd_x = []  # x-data
        self.pd_z = []  # z-data
        self.pd_t = []  # time-data
        self.pd_l = []  # label
        self.pd_f = []  # filename
        self.pd_m = []  # mask (used to mask plotted data)
        self.pd_d = []  # dict, plot description

        # plotted data
        # add one dict for each plotted data
        self.plotted_data = []

        # for movie creation
        self.fps = fps
        self.tscale = tscale
        self.tlabel = tlabel
        self.frames = frames
        self.movie = movie

    def reset_mc_cc(self):
        ''' Reset the marker and color cycle iterators. '''
        self.mc = cerman_rc_params.marker_cycler()()  # invoke the iterator
        self.cc = cerman_rc_params.color_cycler()()
        # self.desc should have a color, even if `colors=off`
        self.desc.update(color=next(self.cc)['color'])
        self.cc = cerman_rc_params.color_cycler()()

    def _clear_fig(self):
        '''Delete any axis and clear figure. '''
        if self.axes is not None:
            for ax in self.axes:
                self.fig.delaxes(ax)
        if self._axes_ghost is not None:
            for ax in self._axes_ghost:
                self._fig_ghost.delaxes(ax)
        if self.fig is not None:
            self.fig.clf()
        if self._fig_ghost is not None:
            self._fig_ghost.clf()

    def _set_fig(self):
        '''Create figure and add axes.'''
        self.reset_mc_cc()      # reset colors
        self._clear_fig()       # clear any present figure

        # set-up dummy figure (for unused axes)
        # note: plotters with more axes set _fig_ghost and _axes_ghost here

        self.fig = Figure()  # create new figure
        gs = GridSpec(1, 1, figure=self.fig)
        self.ax_xz = self.fig.add_subplot(gs[0])   # add axis
        self.axes = [self.ax_xz]                   # store axis list

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def add_data(self, data):
        ''' Parse data, append data to plot.
        Set fpath, llabel, and update origin, as needed.

        Parameters
        ----------
        data :  dict
                loaded simulation data
        '''
        self.set_fpath(fpath=self.fpath, data=data)
        self.set_llabel(data)
        # get data
        x, z, t = self.get_xzt(data)
        x = x * self.xscale
        z = z * self.zscale
        t = t if len(t) == 0 else t * self.tscale
        x, z = self.update_origin(x, z)
        # append data - to be plotted in the end
        self.pd_x.append(x)
        self.pd_z.append(z)
        self.pd_t.append(t)
        self.pd_l.append(self.desc['label'])
        self.pd_d.append(self.desc.copy())
        self.pd_f.append(data['header']['sim_input']['name'])

        msg = 'Added {} data points for: {}'
        logger.debug(msg.format(x.size, self.desc['label']))

    def set_fpath(self, fpath=None, data=None):
        '''Set fpath. Use fpath or get from data, literally.'''

        if fpath is not None:
            self.fpath = fpath
        elif data is not None:
            head = self.fpath_head
            self.fpath = tools.get_fpath(
                data, head=head, folder=self.folder, ext=self.fpath_ext)
        logger.log(5, f'Set fpath: {self.fpath}')

    def set_llabel(self, data):
        '''Use simulation input to set legend label. '''
        label = ''
        lkw = self.legend_lkw.copy()  # copy so name may be removed
        if 'name' in lkw:    # get filename
            label += ', ' + data['header']['sim_input']['name']
            lkw.remove('name')
        if 'sname' in lkw:    # get filename
            label += ', ' + data['header']['save_spec_to_dict']['key']
            lkw.remove('sname')
        if lkw:              # get value strings
            label += ', ' + tools.get_key_data(
                data, keys=lkw, fmt=self.legend_fmt)
        # set label, remove first comma
        label = label[2:]
        logger.log(5, 'Setting label: ' + label)
        self.desc['label'] = label
        return label

    def get_xzt(self, data):
        '''Get plot data. '''
        # this method need to be defined in sub-class
        raise NotImplementedError('Plotter need to be sub-classed')

    def update_origin(self, x, z):
        ''' Offset plot by changing origin. '''
        # allow plotting of nothing, then do nothing
        # to ensure correct legend sequence
        if len(x) != 0:
            # update positions
            x = x + self.originx
            z = z + self.originz
            # update origin after
            self.originx = self.originx + self.diffx
            self.originz = self.originz + self.diffz
        return x, z

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage plotting methods (plot all data)                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def plot_all_pd(self):
        ''' Plot all data in `self.pd_*` to the axis.

        Plot in given order, or sort by legend, as specified.
        Remove non-finite elements.
        Apply the mask in `pd_m` (used for e.g. plotting movie).
        Save the actual plotted data, for dumping later, if specified.
        '''
        def _argsort(seq):
            # return indexes of the sorted sequence.
            sbv = sorted((v, i) for (i, v) in enumerate(seq))
            return [i for (v, i) in sbv]

        # define methods for sorting
        # idea: add other ways to sort
        # idea: allow user to chose sorting
        ksa = list(range(len(self.pd_f)))   # added
        ksf = _argsort(self.pd_f)           # filename
        ksl = _argsort(self.pd_l)           # legend

        # set sorting method (key-index'es)
        kis = ksf
        if self.legend_sorted:
            kis = ksl

        # plot in correct order
        for ki in kis:  # key-index

            # use masked values, if specified
            if self.pd_m:
                mask = self.pd_m[ki]
            else:
                mask = np.isfinite(self.pd_x[ki])

            # get the time-series
            if sum(len(t) for t in self.pd_t) == 0:
                t = []
            elif len(self.pd_t[ki]) == 0:
                t = []
            else:
                t = self.pd_t[ki][mask]

            # get correct data and plot
            x = self.pd_x[ki][mask]
            z = self.pd_z[ki][mask]
            d = self.pd_d[ki]
            self.desc.update(d)
            label = self.plot_xz(x, z)  #
            self.append_plotted_data(x, z, t, label)

        # tod: add this as proper function
        # add fx
        if self.fx is not None:
            # just get extents instead?
            xmin, xmax, zmin, zmax = self.calc_limits()
            x = np.linspace(xmin, xmax, num=100)
            z = np.array([
                sum([ai * xj**i for i, ai in enumerate(self.fx)])
                for xj in x
                ])
            self.curve(x, z)

    def append_plotted_data(self, x, y, t, label):
        ''' Append the plotted data to a dict as strings. '''
        pd = OrderedDict()
        pd['title'] = str(self.title)  # str to ensure copy
        pd['l_lab'] = str(label)
        pd['x_lab'] = str(self.xlabel)
        pd['y_lab'] = str(self.zlabel)
        pd['t_lab'] = str(self.tlabel)
        pd['x'] = str(np.array(x).tolist())
        pd['y'] = str(np.array(y).tolist())
        pd['t'] = str(np.array(t).tolist())
        self.plotted_data.append(pd)

    def save_plotted_data(self):
        ''' Save plotted data to file. '''
        fpath = str(Path(self.fpath).with_suffix('.json'))
        tools.json_dump(self.plotted_data, fpath)
        logger.info(f'Saved {fpath}')

    def plot_xz(self, x, z):
        ''' Actually plot data to the axis, return label.

        Update plot description.
        Plot legend.
        Invoke any/all plotting method(s) specified.
        '''
        # todo: consider making (x, z) into kwargs for generality
        # update descriptions
        if self.colors:
            self.desc.update(color=next(self.cc)['color'])
        if self.markers:
            self.scatter_desc.update(marker=next(self.mc)['marker'])

        # plot the legend label
        desc = self.add_label_to_axes()
        label = desc['label']

        # plot nothing if nothing is found
        # ensures correct legend sequence
        if len(x) == 0:
            logger.debug('Nothing to plot.')
            return label

        # use same description for all these
        if self.curve_on:
            self.curve(x, z)
        if self.scatter_on:
            self.scatter(x, z)
        if self.lowess_on:
            self.lowess(x, z)
        if self.interpolate_on:
            self.interpolate(x, z)
        if self.errorbar_on:
            self.errorbar(x, z)
        if self.minmaxbar_on:
            self.minmaxbar(x, z)
        if self.minmaxfill_on:
            self.minmaxfill(x, z)

        if self.lowess_zx_on:
            self.lowess_zx(x, z)
        if self.interpolate_zx_on:
            self.interpolate_zx(x, z)

        return label

    def update_min_max(self, x, z):
        '''Update plot limits. '''
        # idea: check if this can just be stolen from matplotlib
        self._xmax = max(self._xmax, x.max())
        self._zmax = max(self._zmax, z.max())
        self._xmin = min(self._xmin, x.min())
        self._zmin = min(self._zmin, z.min())

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Plot to given axis                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def add_label_to_axes(self, desc=None):
        ''' Add legend label to all axes. Control visibility elsewhere. '''
        desc = desc or self.desc
        desc = dict(desc)  # to copy
        desc.pop('edgecolor', None)  # key only valid for scatter
        desc.pop('s', None)          # key only valid for scatter
        if self.markers:
            desc['marker'] = self.scatter_desc['marker']
        for ax in self.axes:
            ax.plot([], [], **desc)
        return desc

    def _ax_curve(self, ax, x, z, desc=None):
        '''Add curve plot. '''
        logger.log(5, self._ax_curve.__doc__)
        desc = desc or dict(self.desc, **self.curve_desc)
        ax.plot(x, z, **desc)
        self.update_min_max(x, z)

    @staticmethod
    def _sparsify(x, z, n=100):
        ''' Grid data into a regular nxn grid.
            Return the average position in each group.
        '''
        if len(x) < n**2:
            logger.debug('Small length, skipping sparsify')
            return x, z

        def _bin_idx(x):
            bl = (max(x) - min(x)) / n      # bin length
            left = min(x) - bl              # padding
            right = max(x) + bl
            xf = (x - left) / (right - left)
            return np.floor(xf * n)

        # index values
        xi = _bin_idx(x)
        zi = _bin_idx(z)

        # sort by xi then zi
        xzi = xi * n**2 + zi
        idx_as = np.argsort(xzi)
        xs = x[idx_as]
        zs = z[idx_as]

        # get grouping by looking at grouping in zi
        zg = zi[idx_as]
        zg = np.insert(zg, 0, 0)    # predecessor of first is 0
        zg = zg[0:-1] != zg[1:]     # equal predecessor?
        zg = np.cumsum(zg)          # give each group a number
        zg -= zg[0]                 # ensure start on 0

        # calculate average in each group
        xo = np.array([np.average(xs[zg == i]) for i in range(zg[-1])])
        zo = np.array([np.average(zs[zg == i]) for i in range(zg[-1])])
        logger.debug(f'Sparsified from {len(x)} to {len(xo)}')
        return xo, zo

    def _ax_scatter(self, ax, x, z, desc=None):
        '''Add scatter plot. '''
        logger.log(5, self._ax_scatter.__doc__)
        desc = desc or dict(self.desc, **self.scatter_desc)

        if self.sparsify_on:
            x, z = self._sparsify(x, z, n=self.sparsify_no)

        ax.scatter(x, z, **desc)
        self.update_min_max(x, z)

    def _ax_lowess(self, ax, x, z, desc=None, frac=None):
        '''Add lowess smoothing to plot. '''
        logger.log(5, self._ax_lowess.__doc__)
        desc = desc or dict(self.desc, **self.lowess_desc)
        frac = frac or self.lowess_frac
        nnx = x[~np.isnan(x)]  # non-nan x
        if np.unique(nnx).size < 10:
            logger.info('Skipping lowess, too few datapoints.')
            return
        if self.logz_on:
            z = np.log(z)
        # note: lowess can give a warning if there are too few neighbors
        #       https://github.com/statsmodels/statsmodels/issues/2449
        with warnings.catch_warnings(record=True) as w:
            # note, input to lowess is (val, point), so (z, x) is correct
            out = smoothers_lowess.lowess(z, x, frac=frac, return_sorted=True)
            if w:
                logger.info('Data point issues with lowess.')
            for wi in w:
                logger.log(5, wi)
        (x, z) = (out[:, 0], out[:, 1])
        if self.logz_on:
            z = np.exp(z)
        ax.plot(x, z, **desc)
        self.update_min_max(x, z)

    def _ax_lowess_zx(self, ax, x, z, desc=None, frac=None):
        '''Add lowess smoothing to plot. Use x = x(z).'''
        logger.log(5, self._ax_lowess.__doc__)
        desc = desc or dict(self.desc, **self.lowess_desc)
        frac = frac or self.lowess_frac
        nnz = z[~np.isnan(z)]  # non-nan z
        if np.unique(nnz).size < 10:
            logger.info('Skipping lowess, too few datapoints.')
            return
        if self.logx_on:
            x = np.log(x)
        # note: lowess can give a warning if there are too few neighbors
        #       https://github.com/statsmodels/statsmodels/issues/2449
        with warnings.catch_warnings(record=True) as w:
            # note, input to lowess is (val, point), so (x, z) is correct
            out = smoothers_lowess.lowess(x, z, frac=frac, return_sorted=True)
            if w:
                logger.info('Data point issues with lowess.')
            for wi in w:
                logger.log(5, wi)
        (z, x) = (out[:, 0], out[:, 1])
        if self.logx_on:
            x = np.exp(x)
        ax.plot(x, z, **desc)
        self.update_min_max(x, z)

    @staticmethod
    def _calc_stat(x, z, sort=True):
        ''' Calculate statistics for each unique x-value.

        Parameters
        ----------
        x : float
            x-values
        z : float
            z-values

        Returns
        -------
        xu: float
            x-values, unique, sorted
        za: float
            z-values, average
        zs: float
            z-values, standard deviation
        zmin: float
            z-values, minimum
        zmax: float
            z-values, maximum
        '''
        xu = np.unique(x)
        if sort:
            xu = np.array(sorted(xu))
        za = np.array([np.average(z[x == xi]) for xi in xu])
        zs = np.array([np.std(z[x == xi]) for xi in xu])
        zmin = np.array([np.min(z[x == xi]) for xi in xu])
        zmax = np.array([np.max(z[x == xi]) for xi in xu])
        return xu, za, zs, zmin, zmax

    def _ax_interpolate(self, ax, x, z, desc=None):
        '''Add interpolation to plot. '''
        logger.log(5, self._ax_interpolate.__doc__)
        desc = desc or dict(self.desc, **self.interp_desc)
        if np.unique(x).size < 2:
            msg = 'Skipping interpolate, too few datapoints ({}).'
            logger.info(msg.format(np.unique(x)))
            return
        xu, za, zs, zmin, zmax = self._calc_stat(x, z, sort=True)
        xl = np.linspace(min(xu), max(xu), 500)
        za = pchip_interpolate(xu, za, xl)
        ax.plot(xl, za, **desc)
        self.update_min_max(xl, za)

    def _ax_interpolate_zx(self, ax, x, z, desc=None):
        '''Add interpolation to plot. Use x = x(z).'''
        logger.log(5, self._ax_interpolate.__doc__)
        desc = desc or dict(self.desc, **self.interp_desc)
        if np.unique(z).size < 2:
            msg = 'Skipping interpolate, too few datapoints ({}).'
            logger.info(msg.format(np.unique(z).size))
            return
        zu, xa, xs, xmin, xmax = self._calc_stat(z, x, sort=True)
        zl = np.linspace(min(zu), max(zu), 500)
        xa = pchip_interpolate(zu, xa, zl)
        ax.plot(xa, zl, **desc)
        self.update_min_max(xa, zl)

    def _ax_errorbar(self, ax, x, z, desc=None):
        '''Add errorbars to plot. '''
        logger.log(5, self._ax_errorbar.__doc__)
        desc = desc or dict(self.desc, **self.scatter_desc)
        desc.pop('s', None)
        xu, za, zs, zmin, zmax = self._calc_stat(x, z, sort=True)
        ax.errorbar(xu, za, yerr=zs * 2, fmt='o', **desc)
        self.update_min_max(x, z)

    def _ax_minmaxbar(self, ax, x, z, desc=None):
        '''Add minmaxbars to plot. '''
        logger.log(5, self._ax_minmaxbar.__doc__)
        desc = desc or dict(self.desc, **self.scatter_desc)
        desc.pop('s', None)
        xu, za, zs, zmin, zmax = self._calc_stat(x, z, sort=True)
        zerr = [za - zmin, zmax - za]
        ax.errorbar(xu, za, yerr=zerr, fmt='o', **desc)
        self.update_min_max(x, z)

    def _ax_minmaxfill(self, ax, x, z, desc=None):
        '''Add fill min-max to plot. '''
        logger.log(5, self._ax_minmaxfill.__doc__)
        desc = desc or dict(self.desc, **self.interp_desc)
        desc.pop('ls', None)        # invalid key
        desc.pop('marker', None)    # invalid key
        desc.pop('edgecolor', None) # invalid key
        desc.pop('s', None)         # invalid key
        xu, za, zs, zmin, zmax = self._calc_stat(x, z, sort=True)
        xl = np.linspace(min(xu), max(xu), 500)
        za = pchip_interpolate(xu, za, xl)
        zmin = pchip_interpolate(xu, zmin, xl)
        zmax = pchip_interpolate(xu, zmax, xl)
        diff = (zmax - zmin) / 20 * 0  # center space?
        ax.fill_between(xl, zmax, za - diff, alpha=.15, **desc)
        ax.fill_between(xl, za + diff, zmin, alpha=.15, **desc)
        self.update_min_max(xl, za)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Plot to all axes                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def curve(self, x, z, desc=None):
        ''' Add curve plot to all axes. '''
        self._ax_curve(self.ax_xz, x, z, desc=desc)

    def scatter(self, x, z, desc=None):
        ''' Add scatter plot to all axes. '''
        self._ax_scatter(self.ax_xz, x, z, desc=desc)

    def lowess(self, x, z, desc=None):
        ''' Add lowess plot to all axes. '''
        self._ax_lowess(self.ax_xz, x, z, desc=desc)

    def lowess_zx(self, x, z, desc=None):
        ''' Add lowess_zx plot to all axes. '''
        self._ax_lowess_zx(self.ax_xz, x, z, desc=desc)

    def interpolate(self, x, z, desc=None):
        ''' Add interpolate plot to all axes. '''
        self._ax_interpolate(self.ax_xz, x, z, desc=desc)

    def interpolate_zx(self, x, z, desc=None):
        ''' Add interpolate_zx plot to all axes. '''
        self._ax_interpolate_zx(self.ax_xz, x, z, desc=desc)

    def errorbar(self, x, z):
        ''' Add errorbar plot to all axes. '''
        self._ax_errorbar(self.ax_xz, x, z)

    def minmaxbar(self, x, z, desc=None):
        ''' Add minmaxbar plot to all axes. '''
        self._ax_minmaxbar(self.ax_xz, x, z, desc=desc)

    def minmaxfill(self, x, z, desc=None):
        ''' Add minmaxfill plot to all axes. '''
        self._ax_minmaxfill(self.ax_xz, x, z, desc=desc)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Finalize plot                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def calc_limits(self):
        ''' Return plot limits.
        Use the min/max of the added data.
        Make symmetric, if specified.
        Use ratio, if specified.
        '''
        # calculate plot limits

        # set symmetric
        if self.symx:
            self._xmax = max(-self._xmin, self._xmax)
            self._xmin = -self._xmax
        if self.symz:
            self._zmax = max(-self._zmin, self._zmax)
            self._zmin = -self._zmax

        # pre-calculation
        dx = self._xmax - self._xmin  # x-distance
        dz = self._zmax - self._zmin
        ax = (self._xmax + self._xmin) / 2  # average
        az = (self._zmax + self._zmin) / 2  # average

        if self.xz_ratio:
            # set z/x ratio
            # note, this only works when the axes are equal
            r = self.xz_ratio
            # keep the below as a single operation
            (dx, dz) = (max(dx, dz / r), max(dz, dx * r))

        # add padding (after correcting ratio)
        f = 0.1     # add this fraction to plot limit
        px = dx * f                   # x-padding
        pz = dz * f

        # keep min = 0, change other
        xmin = self._xmin
        zmin = self._zmin
        if xmin != 0:
            xmin = ax - (dx + px) / 2
        if zmin != 0:
            zmin = az - (dz + pz) / 2
        # set max relative to min
        xmax = xmin + px + dx
        zmax = zmin + pz + dz

        logger.debug(f'Limits calc: x=({xmin},{xmax}), z=({zmin},{zmax})')
        return xmin, xmax, zmin, zmax

    def set_limits(self):
        ''' Set axes lin/log and set limits.
            Set calculated limits, then override with user defined values.
            Note: xlim is applied for all the axes, just at first.
        '''
        logger.debug('Setting limits')
        # todo: consider adding scientific notation as option
        #       ax.ticklabel_format(style=sci, scilimits=(0, 0))

        # set equal limits for all axes
        xmin, xmax, zmin, zmax = self.calc_limits()

        if self.logx_on:
            xmin = None if (xmin <= 0) else xmin
            xmax = None if (xmax <= 0) else xmax
            for ax in self.axes:
                ax.set_xscale('log')
        if self.logz_on:
            zmin = None if (zmin <= 0) else zmin
            zmax = None if (zmax <= 0) else zmax
            for ax in self.axes:
                ax.set_yscale('log')

        # if min==max, then do not override Matplotlib's default
        if (xmin == xmax):
            (xmin, xmax) = (None, None)
        if (zmin == zmax):
            (zmin, zmax) = (None, None)

        for ax in self.axes:
            # set calculated limits
            # sub-classes with more axes should use xmin/xmax as default
            ax.set_xlim((xmin, xmax))   # xlim may vary between axes
            ax.set_ylim((zmin, zmax))   # ylim to be equal for all axes
            # override with user defined values
            # note: warning is fine if user specifies negative values with log
            ax.set_xlim(left=self.xmin, right=self.xmax)
            ax.set_ylim(bottom=self.zmin, top=self.zmax)

        xt, zt = self.axes[0].get_xlim(), self.axes[0].get_ylim(),
        logger.debug(f'Limits set: x=({xt}), z=({zt})')

    def set_annotation(self):
        ''' Annotate axis, if specified. '''
        if not self.annotate_on:
            return
        logger.debug('Adding annotation')
        # idea: add possibility to fix axis
        # idea: add possibility to fix style
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        bbox_props = dict(boxstyle="circle", fc="w", ec="0.5", alpha=0.9)
        self.axes[0].text(
            self.annotate_x,
            self.annotate_y,
            self.annotate_text,
            ha="center",
            va="center",
            bbox=bbox_props
            )

    def set_labels(self):
        '''Control legend, title, and axis labels. '''
        logger.debug('Setting labels')
        if self.legend_on:
            # add to only one axis, -1 for furthest right
            self.axes[-1].legend(numpoints=1, loc=self.legend_loc)
        if self.xlabel:
            self.ax_xz.set_xlabel(self.xlabel)
        if self.zlabel:
            # add to only one axis, 0 for furthest left
            self.axes[0].set_ylabel(self.zlabel)
        if self.title == 'fpath':
            self.title = str(Path(self.fpath).stem)
        if self.title:
            self.fig.suptitle(self.title)
        if self.annotate_on:
            self.set_annotation()

    def save_plot(self, close=True):
        ''' Save added data as plotted figure, movie, and/or json-file.
        Create figure, add all plots, set limits, set labels, save figure.
        '''
        if self.fpath is None:
            logger.error('Error: cannot save plot without name.')
            return
        self._set_fig()                 # create the figure
        self.plot_all_pd()              # plot to figure
        self.set_limits()
        self.set_labels()
        self.fig.set_tight_layout(self.tight)
        logger.info('Saving: {}'.format(self.fpath))
        if self.figure:
            Canvas(self.fig).print_figure(self.fpath)
        if self.movie:
            self.save_movie()
        if close:
            self._clear_fig()
        if self.save_data:
            self.save_plotted_data()

    def save_movie(self):
        '''Save a movie of a plot with a time-series.

        Note:
        matplotlib.animation is imported below. FFMpegWriter is used.
        This method solved problems when running in background.

        Calculate time position of each frame.
        tscale=1e6 implies that 1 us takes 1 s.
        fps=5 implies 5 frames per second.
        '''

        # assume that everything is plotted
        # that axes and labels are correct
        # idea: add option to use no or idx instead of time

        if (len(self.pd_t) == 0) or sum(len(t) for t in self.pd_t) == 0:
            msg = 'Unable to create movie. No time data.'
            logger.info(msg)
            return

        # temporary change rcParams
        # (since tight bbox breaks ffmpeg writer)
        # ffmpeg_file could be used, but is possibly slower
        # self.fig.set_tight_layout(True)
        bbox = mpl.rcParams['savefig.bbox']
        # mpl.rcParams['savefig.bbox'] = 'standard'

        # get min/maximum/diff time
        # assume t is scaled correctly, filter empty series
        tmax = max(t.max() for t in self.pd_t if len(t) > 0)
        tmin = min(t.min() for t in self.pd_t if len(t) > 0)
        tdiff = tmax - tmin

        # calc number of frames
        # (10 us * 1e7(tscale) * 5 f/s = 500 frames)
        frame_no = int(tdiff * self.fps)
        logger.info('Creating movie of {} frames.'.format(frame_no))
        logger.log(5, 'tmax * self.fps, ' + str(tmax * self.fps))
        logger.debug('Max time: {}'.format(tmax))

        # set up linspace of which times to plot
        fts = np.linspace(tmin, tmax, frame_no, endpoint=True)

        # configure writer
        import matplotlib.animation as mpla  # hack: see comment on top
        FFMpegWriter = mpla.FFMpegWriter      # pipe based
        # FFMpegWriter = mpla.FFMpegFileWriter  # writing temp file
        metadata = dict(title=self.title, artist='CerMan')
        writer = FFMpegWriter(fps=self.fps, metadata=metadata)
        dpi = mpl.rcParams['savefig.dpi']
        vpath = str(Path(self.fpath).with_suffix('.mp4'))
        # note: do note clear&set figure after this point!
        writer.setup(fig=self.fig, outfile=vpath, dpi=dpi)

        # choose at which iterations to log
        log_i = [int(frame_no * j / 10) for j in range(1, 10)]

        # create movie frames
        for i, ft in enumerate(fts):
            # logging
            if i in log_i:
                msg = 'Creating movie {:4.1f} %'
                logger.info(msg.format(i / frame_no * 100))
            msg = 'Plotting frame {:d} of {:d}'
            logger.log(5, msg.format(i + 1, frame_no))

            # todo: consider skipping of reset/redraw
            #       if there is no new info (sum(t<ft), now vs last)

            # reset plot
            for ax in self.axes:
                ax.clear()
            self.reset_mc_cc()

            # add time to legend
            tlabel = f'Time: {ft:6.3f} {self.tlabel}'
            self.add_label_to_axes(desc={'label': tlabel, 'alpha': 0})

            # set mask and plot a frame
            self.pd_m = [ti < ft for ti in self.pd_t]  # empty when empty, ok
            self.plot_all_pd()
            self.set_limits()
            self.set_labels()

            Canvas(self.fig).draw()  # redraw figure needed
            writer.grab_frame()      # grab frame for video

            # save individual frames
            if self.frames:
                path = Path(self.fpath).with_suffix('')
                L = len(str(frame_no))  # pad length
                path = str(path) + '{0:0{1}d}'.format(i, L)
                logger.info('Saving frame: {}'.format(path))
                Canvas(self.fig).print_figure(path)

        # save video
        logger.info('Saving: {}'.format(vpath))
        writer.finish()

        # rewind setting
        mpl.rcParams['savefig.bbox'] = bbox


class ExamplePlotter(PlotterXZ):
    '''EXAMPLE: Plot simulation time against iteration number.'''

    def __init__(self, title=None, scatter=None,
                 zlabel=None, zscale=None, xlabel=None, xscale=None,
                 **kwargs):
        super().__init__(**kwargs)  # send kwargs to super
        # override super, and set special, as needed
        self.fpath_head = 'example'
        self.title = 'Simulation Time' if title is True else title
        self.xlabel = 'Iteration Number' if xlabel is None else xlabel
        self.zlabel = 'Time [\u00B5s]' if zlabel is None else zlabel
        self.xscale = 1e0 if xscale is None else xscale
        self.zscale = 1e6 if zscale is None else zscale
        self.scatter_on = True if scatter is None else scatter

    def get_xzt(self, data):
        x = np.array(data.get('no'))
        z = np.array(data.get('sim_time'))
        t = z
        return x, z, t


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       JSON PLOTTER                                                  #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class JsonPlotter(PlotterXZ):

    @classmethod
    def _get_opt_string(cls):
        opts = super()._get_opt_string()
        # opts['fpath'] = 'Path of json-file to create plot from'
        return opts

    @classmethod
    def _get_opt_bool(cls):
        opts = super()._get_opt_bool()
        # opts['plot_zx'] = 'Reverse axis for lowess, plot x(z).'
        return opts

    def __init__(self, title=None, scatter=None,
                 xscale=1, zscale=1, tscale=1,
                 zlabel=None, xlabel=None, tlabel=None,
                 **kwargs):
        super().__init__(**kwargs)

        # idea: should probably default to loaded data if true and that exists
        self.title = 'fpath' if title is True else title
        self.xlabel = xlabel
        self.zlabel = zlabel
        self.tlabel = tlabel
        self.scatter_on = True if scatter is None else scatter
        self.xscale = xscale  # not be scaled by default
        self.zscale = zscale  # not be scaled by default
        self.tscale = tscale  # not be scaled by default

    def add_data(self, fpath):
        ''' Load json-data and add to plot data.

        Parameters
        ----------
        data :  list[dict]
                loaded json-formatted data
        '''

        # load data
        data = tools.json_load(fpath)

        # set fpath, if needed
        if self.fpath is None:
            self.fpath = tools.tools.get_fpath(
                fpath, head=self.fpath_head,
                folder=self.folder, ext=self.fpath_ext)

        # unpack data
        for pd in data:
            # allow these to be overruled by class initiation
            if self.title is None:
                self.title = pd['title']
            if self.xlabel is None:
                self.xlabel = pd['x_lab']
            if self.zlabel is None:
                self.zlabel = pd['y_lab']  # hack for historic reasons
            if self.tlabel is None:
                if 't_lab' in pd:
                    self.tlabel = pd['t_lab']
                else:
                    self.tlabel = ''
            self.desc['label'] = pd['l_lab']

            # data is saved as stings, so this looks a bit weird
            x = np.fromstring(pd['x'][1:-1], sep=', ')
            z = np.fromstring(pd['y'][1:-1], sep=', ')  # hack
            if 't' in pd and pd['t']:
                t = np.fromstring(pd['t'][1:-1], sep=', ')
            else:
                t = []

            # remove 'nan' from data
            mask = np.isnan(x) | np.isnan(z)
            x = x[~mask] * self.xscale
            z = z[~mask] * self.zscale
            if len(t) > 0:
                t = t[~mask] * self.tscale
            # change origin (if diffx/z)
            x, z = self.update_origin(x, z)
            # append the data series
            self.pd_x.append(x)
            self.pd_z.append(z)
            self.pd_t.append(t)
            self.pd_l.append(pd['l_lab'])
            self.pd_d.append(self.desc.copy())
            self.pd_f.append(fpath)

            msg = 'Added {} data points for: {}'
            logger.debug(msg.format(x.size, pd['l_lab']))

    # the methods below are not meant to be implemented
    def set_fpath(self, fpath):
        raise NotImplementedError

    def set_llabel(self, data):
        raise NotImplementedError

    def get_xzt(self, data):
        raise NotImplementedError


# store a dict of available plotters
plotters = {
    'example': ExamplePlotter,
    'json': JsonPlotter,
}


#
