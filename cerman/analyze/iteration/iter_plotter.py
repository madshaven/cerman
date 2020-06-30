#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Superclass for plotting data from a given iteration.
'''

# Imports for data manipulation
import numpy as np
import logging
from pathlib import Path

# Imports for plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib import ticker

# Import from project files
from .. import tools
from .. import plotterXYRZ

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE PLOTTER XYRZ                                           #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class IterPlotter(plotterXYRZ.PlotterXYRZ):
    '''Superclass for plotting data from one iteration.

    The `plot_descs` describes which data to extract.
    The dict key is the `save_key` to use.
    The actual dict describes how to plot.
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # the plot description list defines which keys to plot
        self.plot_descs = {
            # 'save_key': {'label': 'default'},  # base case
            }

        self.symx = True  # symmetric x-axis

        # plot setup
        self.scatter_desc = {'s': 5, 'zorder': 5, 'marker': '.',
                             'alpha': 0.5, 'facecolors': 'none',
                             'label': None}

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def add_data_for_key(self, data, plot_key):
        ''' Add plot data for the provided key. '''
        x, y, r, z = self.get_xyrz(data, plot_key)
        x = x * self.xscale
        y = y * self.yscale
        r = r * self.rscale
        z = z * self.zscale
        # x, y, r, z = self.update_origin(x, y, r, z)
        self.update_desc(plot_key)
        self.pd_x.append(x)
        self.pd_y.append(y)
        self.pd_r.append(r)
        self.pd_z.append(z)
        # self.pd_t.append(t)
        self.pd_l.append(self.desc['label'])
        self.pd_d.append(self.desc.copy())
        self.pd_f.append(data['header']['sim_input']['name'])

    def add_data(self, data, idx=None, no=None):
        ''' Parse data, append data to plot.
        Set fpath, llabel, and update origin, as needed.

        Parameters
        ----------
        data :  dict
                loaded simulation data
        '''
        # Override 'idx' if 'no' is specified
        if no is not None:
            idx = data['no'].index(no)
        self.data = data
        self.idx = idx
        # extract label from plot_descs, not from given data
        # self.set_llabel(data)  # to be ignored
        # idea: use llabel to set title for iteration plots
        self.set_fpath(fpath=self.fpath, data=self.data)
        for plot_key in self.plot_descs:
            self.add_data_for_key(self.data, plot_key)

    def update_desc(self, plot_key):
        self.desc.update(self.plot_descs[plot_key])

    def get_xyrz(self, data, plot_key):
        '''Get plot data. '''
        # this method need to be defined in sub-class
        raise NotImplementedError('Plotter need to be sub-classed')

    def dirty_get(self, data, key, idx):
        # try to get data, return empty coordinate if exception occurs
        try:
            value = data.get(key)[idx]
        except:
            logger.log(5, 'Could not get {}, skipping...'.format(key))
            value = np.array([]).reshape(3, -1)
        return value

    def set_fpath(self, fpath=None, data=None):
        '''Set fpath. Use fpath or get from data, literally.'''

        if fpath is not None:
            self.fpath = fpath

        elif data is not None:
            stem = Path(tools.get_fpath(data)).stem
            folder = self.folder + '_' + stem
            no = data['no'][self.idx]
            max_no = data['no'][-1]
            tail = tools.padded_no(no, no_max=max_no)
            fpath = tools.get_fpath(
                data,
                head=self.fpath_head,
                folder=folder,
                ext=self.fpath_ext,
                tail=tail)
            self.fpath = fpath


#
