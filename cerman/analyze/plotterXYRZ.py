#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Class for plotting an XZ- , an RZ and an YZ-axis in the same plot.
    See PlotterXZ for general info.
'''

# General imports
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import numpy as np
import logging

# Import from project files
from . import plotterXZ

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE PLOTTER XYZ                                            #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class PlotterXYRZ(plotterXZ.PlotterXZ):

    @classmethod
    def _get_opt_float(cls):
        opts = super()._get_opt_float()
        opts['ymin'] = 'Set y-axis minimum'
        opts['ymax'] = 'Set y-axis maximum'
        opts['rmin'] = 'Set r-axis minimum'
        opts['rmax'] = 'Set r-axis maximum'
        opts['yscale'] = 'Set scale of y-axis'
        opts['rscale'] = 'Set scale of r-axis'
        opts['scale'] = 'Set scale of all axes'
        opts['xz_ratio'] = 'Set xz-ratio for all axes'
        opts['diffx'] = 'Set offset in x&y&r for each plot'
        opts['diffy'] = 'Set offset in y for each plot'
        opts['diffr'] = 'Set offset in r for each plot'
        opts['diffz'] = 'Set offset in z for each plot'
        return opts

    @classmethod
    def _get_opt_string(cls):
        opts = super()._get_opt_string()
        opts.pop('fx', None)  # not valid
        opts['ylabel'] = 'Set y-axis label'
        opts['rlabel'] = 'Set r-axis label'
        opts['axes_visible'] = 'Choose axes to plot `xyr`'
        return opts

    def __init__(self,
                 # control axis
                 rlabel='R', rscale=1, rmin=None, rmax=None,
                 ylabel='Y', yscale=1, ymin=None, ymax=None,
                 diffx=0, diffy=0, diffr=0, diffz=0,
                 scale=None, axes_visible='xyr', xz_ratio=2.2,
                 **kwargs):
        super().__init__(**kwargs)
        # get input / set defaults / sub-class to override if needed
        self.rlabel = rlabel
        self.ylabel = ylabel
        self.rscale = rscale
        self.yscale = yscale
        if scale is not None:
            self.xscale = scale
            self.yscale = scale
            self.rscale = scale
            self.zscale = scale

        # plot limits, set by initiation
        # leave as 'none' to use plotted data instead
        # note: default is to set xy&r-axis equal
        self.ymin = ymin
        self.ymax = ymax
        self.rmin = rmin
        self.rmax = rmax

        # possibly add offset to plots
        self.originx = 0
        self.originy = 0
        self.originr = 0
        self.originz = 0
        self.diffx = diffx
        self.diffy = diffy or diffx
        self.diffr = diffr or diffx
        self.diffz = diffz

        # plot setup
        self.xz_ratio = xz_ratio
        self.axes_visible = axes_visible

        # plot data
        self.pd_y = []  # y-data
        self.pd_r = []  # r-data

        # add one dict for each plotted data for each axis
        self.plotted_data = []

    def _set_fig(self):
        '''Create figure and add axes.'''

        self.reset_mc_cc()      # reset colors
        self._clear_fig()       # clear any present figure

        # set-up dummy figure (for unused axes)
        self._fig_ghost = Figure()
        gs = GridSpec(1, 3, figure=self.fig)
        self.ax_xz = self._fig_ghost.add_subplot(gs[0], aspect='equal')
        self.ax_yz = self._fig_ghost.add_subplot(gs[1], aspect='equal')
        self.ax_rz = self._fig_ghost.add_subplot(gs[2], aspect='equal')
        self._axes_ghost = [self.ax_xz, self.ax_yz, self.ax_rz]

        # set-up actual figure
        self.fig = Figure()     # create new figure
        self.axes = []
        m = len(self.axes_visible)
        n = 0
        gs = GridSpec(1, m, figure=self.fig)
        for key in self.axes_visible:
            if key == 'x':
                self.ax_xz = self.fig.add_subplot(gs[n], aspect='equal')
                self.axes.append(self.ax_xz)
                n += 1
            if key == 'y':
                self.ax_yz = self.fig.add_subplot(gs[n], aspect='equal')
                self.axes.append(self.ax_yz)
                n += 1
            if key == 'r':
                self.ax_rz = self.fig.add_subplot(gs[n], aspect='equal')
                self.axes.append(self.ax_rz)
                n += 1

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
        x, y, r, z, t = self.get_xyrzt(data)
        x = x * self.xscale
        y = y * self.yscale
        r = r * self.rscale
        z = z * self.zscale
        t = t if len(t) == 0 else t * self.tscale
        x, y, r, z = self.update_origin(x, y, r, z)
        # append data - to be plotted in the end
        self.pd_x.append(x)
        self.pd_y.append(y)
        self.pd_r.append(r)
        self.pd_z.append(z)
        self.pd_t.append(t)
        self.pd_l.append(self.desc['label'])
        self.pd_d.append(self.desc.copy())
        self.pd_f.append(data['header']['sim_input']['name'])

        msg = 'Added {} data points for: {}'
        logger.debug(msg.format(x.size, self.desc['label']))

    def get_xyrzt(self, data):
        '''Get plot data. '''
        # this method need to be defined in sub-class
        raise NotImplementedError('Plotter need to be sub-classed')

    def update_origin(self, x, y, r, z):
        ''' Offset plot by changing origin. '''
        # allow plotting of nothing
        # ensures correct legend sequence
        if len(x) != 0:
            # update positions
            x = x + self.originx
            y = y + self.originy
            r = r + self.originr
            z = z + self.originz
            # update origin after
            self.originx = self.originx + self.diffx
            self.originy = self.originy + self.diffy
            self.originr = self.originr + self.diffr
            self.originz = self.originz + self.diffz
        return x, y, r, z

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage plotting methods (plot all data)                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def plot_all_pd(self):
        ''' Plot all data in `self.pd_*` to the axes.

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
        for ki in kis:
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
            y = self.pd_y[ki][mask]
            r = self.pd_r[ki][mask]
            z = self.pd_z[ki][mask]
            d = self.pd_d[ki]
            self.desc.update(d)
            label = self.plot_xyrz(x, y, r, z)
            self.append_plotted_data(x, y, r, z, t, label)

    def append_plotted_data(self, x, y, r, z, t, label):
        ''' Append the plotted data to a dict as strings. '''
        if 'x' in self.axes_visible:
            super().append_plotted_data(x, z, t, f'{label}, xz-axis')
        if 'y' in self.axes_visible:
            super().append_plotted_data(y, z, t, f'{label}, yz-axis')
        if 'r' in self.axes_visible:
            super().append_plotted_data(r, z, t, f'{label}, rz-axis')

    def plot_xyrz(self, x, y, r, z):
        ''' Actually plot data to the axes, return label.

        Update plot description.
        Plot legend.
        Invoke any/all plotting method(s) specified.
        '''
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
            return

        if self.curve_on:
            self.curve(x, y, r, z)
        if self.scatter_on:
            self.scatter(x, y, r, z)

        # todo: implement these
        if self.lowess_on:
            logger.warning(f'`lowess` not implemented')
            # self.lowess(x, z)
        if self.interpolate_on:
            logger.warning(f'`interpolate` not implemented')
            # self.interpolate(x, z)
        if self.errorbar_on:
            logger.warning(f'`errorbar` not implemented')
            # self.errorbar(x, z)
        if self.minmaxbar_on:
            logger.warning(f'`minmaxbar` not implemented')
            # self.minmaxbar(x, z)
        if self.minmaxfill_on:
            logger.warning(f'`minmaxfill` not implemented')
            # self.minmaxfill(x, z)

        if self.lowess_zx_on:
            logger.warning(f'`lowess_zx` not implemented')
            # self.lowess_zx(x, z)
        if self.interpolate_zx_on:
            logger.warning(f'`interpolate_zx` not implemented')
            # self.interpolate_zx(x, z)

        return label

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Plot to given axis                                        #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # performed by super
    # these methods also update xmin/xmax, zmin/zmax
    # min/max for r and y is not saved explicitly, but can be set by user

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Plot to all axes                                          #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def scatter(self, x, y, r, z, desc=None):
        self._ax_scatter(self.ax_xz, x, z, desc=desc)
        self._ax_scatter(self.ax_yz, y, z, desc=desc)
        self._ax_scatter(self.ax_rz, r, z, desc=desc)

    def curve(self, x, y, r, z, desc=None):
        self._ax_curve(self.ax_xz, x, z, desc=desc)
        self._ax_curve(self.ax_yz, y, z, desc=desc)
        self._ax_curve(self.ax_rz, r, z, desc=desc)

    # def lowess(self, x, z, desc=None):
    #     self._ax_lowess(self.ax_xz, x, z, desc=desc)

    # def interpolate(self, x, z, desc=None):
    #     self._ax_interpolate(self.ax_xz, x, z, desc=desc)

    # def errorbar(self, x, z):
    #     self._ax_errorbar(self.ax_xz, x, z)

    # def minmaxbar(self, x, z, desc=None):
    #     self._ax_minmaxbar(self.ax_xz, x, z, desc=desc)

    # def minmaxfill(self, x, z, desc=None):
    #     self._ax_minmaxfill(self.ax_xz, x, z, desc=desc)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Finalize plot                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def set_limits(self):
        ''' Set axes lin/log and set limits.
            Set calculated limits, then override with user defined values.
            Note: xlim is applied for all the axes, just at first.
        '''
        super().set_limits()
        # override xlim set by super, if needed
        self.ax_yz.set_xlim(left=self.ymin, right=self.ymax)
        (rmin, rmax) = self.ax_rz.get_xlim()
        dr = rmax - rmin
        rmin = - dr / 20
        rmax = rmin + dr
        self.ax_rz.set_xlim(left=rmin, right=rmax)
        self.ax_rz.set_xlim(left=self.rmin, right=self.rmax)

    def set_labels(self):
        '''Control legend, title, and axes labels.
        Note: super sets zlabel for first axis and legend for last axis.
        '''
        super().set_labels()
        if self.ylabel:
            self.ax_yz.set_xlabel(self.ylabel)
        if self.rlabel:
            self.ax_rz.set_xlabel(self.rlabel)
        for ax in self.axes[1:]:    # exclude the first
            ax.get_yaxis().set_ticklabels([])   # no y-tick labels


#
