#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Functions for scatter plotting of single iteration.
'''

import numpy as np
import logging
from collections import OrderedDict

from ..load_data import LoadedData
from ..cerman_rc_params import cerman_rc_params
from . import iter_plotter

# Settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE PLOTTER                                                #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ScatterPlotter(iter_plotter.IterPlotter):
    '''Superclass for scatter plotting. '''

    @classmethod
    def _get_opt_string(cls):
        opts = super()._get_opt_string()
        opts['colors_dict'] = 'Number of colors, defaults to 12'
        return opts

    def __init__(self, scatter=None, scale=1e3,
                 xlabel=None, ylabel=None, rlabel=None, zlabel=None,
                 colors_dict=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.xlabel = r'Position $x$ [mm]' if xlabel is None else xlabel
        self.ylabel = r'Position $y$ [mm]' if ylabel is None else ylabel
        self.rlabel = r'Position $r$ [mm]' if rlabel is None else rlabel
        self.zlabel = r'Position $z$ [mm]' if zlabel is None else zlabel
        self.colors_dict = '12' if colors_dict is None else colors_dict

        if scale:
            self.xscale = scale
            self.yscale = scale
            self.rscale = scale
            self.zscale = scale

        # note for '.', 10 is quite small, 20 also works
        self.scatter_desc = {'s': 10, 'zorder': 5, 'marker': '.',
                             'alpha': 1, 'label': None}

        # self.scatter_desc = {'s': 50, 'zorder': 5, 'marker': 'o',
        #                      'alpha': 0.75, 'facecolors': 'none'}

        self.scatter_on = True if scatter is None else scatter
        self.legend_sorted = False

    def reset_mc_cc(self):
        ''' Reset the marker and color cycle iterators. '''
        self.mc = cerman_rc_params.marker_cycler()()  # invoke the iterator
        self.cc = cerman_rc_params.color_cycler(self.colors_dict)()
        # self.desc should have a color, even if `colors=off`
        self.desc.update(color=next(self.cc)['color'])
        self.cc = cerman_rc_params.color_cycler(self.colors_dict)()

    def plot_xyrz(self, x, y, r, z):
        ''' Actually plot data to the axes, return label.

        Update plot description.
        Plot legend.
        Invoke any/all plotting method(s) specified.
        '''
        # update descriptions
        color = next(self.cc)['color']
        self.desc.update(edgecolor=color, color=color)
        label = self.add_label_to_axes()

        # plot nothing if nothing is found
        # ensures correct legend sequence
        if len(x) == 0:
            logger.debug('Nothing to plot.')
            return label

        if self.curve_on:
            self.curve(x, y, r, z)
        if self.scatter_on:
            self.scatter(x, y, r, z)

        return label

    def get_xyrz(self, data, key):
        # Get data
        pos = self.dirty_get(data, key, self.idx)
        x = pos[0, :]
        y = pos[1, :]
        z = pos[2, :]
        r = np.sqrt(pos[0, :]**2 + pos[1, :]**2)
        return x, y, r, z


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE SPECIAL PLOTTERS                                       #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class HeadsPlotter(ScatterPlotter):
    '''Scatter plot streamer head heads.'''

    def __init__(self, title=False, folder='heads', **kwargs):
        super().__init__(**kwargs)

        self.title = 'Scatter Heads' if title is True else title
        self.folder = folder

        # the plot description list defines which keys to plot
        pdl = [
               ('streamer_pos', 'Current'),
               ('streamer_pos_appended', 'New'),
               ('streamer_pos_removed', 'Removed'),
               ]
        self.plot_descs = OrderedDict(
            (k, {'label': v}) for (k, v) in pdl)


class SeedsPlotter(ScatterPlotter):
    '''Scatter plot seeds.'''

    def __init__(self, title=False, folder='seeds', **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.folder = folder

        # the plot description list defines which keys to plot
        pdl = [
               ('seeds_pos', 'Position'),
               ('seeds_pos_ion', 'Ion'),
               ('seeds_pos_electron', 'Electron'),
               ('seeds_pos_to_remove', 'Remove'),
               ('seeds_pos_avalanche', 'Avalanche'),
               ('seeds_pos_critical', 'Critical'),
               ('seeds_pos_behind_roi', 'Behind'),
               ('seeds_pos_in_streamer', 'Collided'),
               ('seeds_pos_to_append', 'New'),
               ]
        self.plot_descs = OrderedDict(
            (k, {'label': v}) for (k, v) in pdl)

    def get_xyrz(self, data, key):
        # Get data
        pos = LoadedData(data=self.data).get_seeds(key=key, idx=self.idx)
        x = pos[0, :]
        y = pos[1, :]
        z = pos[2, :]
        r = np.sqrt(pos[0, :]**2 + pos[1, :]**2)
        return x, y, r, z


class SeedHeadPlotter(ScatterPlotter):
    '''Scatter plot seeds and streamer head heads.'''

    def __init__(self, **kwargs):
        kwargs['title'] = kwargs.get('title', None)
        kwargs['folder'] = kwargs.get('folder', 'seedhead')
        super().__init__(**kwargs)

        # higher zorder is on top
        self.seed_desc = {'s': 10, 'marker': '.',
                          'zorder': 5, 'alpha': 1,
                          # 'label': None,
                          }
        self.streamer_desc = {'s': 60, 'marker': 'x', 'lw': 1,
                              'zorder': 6, 'alpha': 1,
                              # 'label': None,
                              }
        # prevent global scatter desc from overwriting
        self.scatter_desc = {'label': None}

        # the plot description list defines which keys to plot
        pdl = [
            # ('seeds_pos', 'Position'),
            # ('seeds_pos_ion', 'Ion'),
            # ('seeds_pos_electron', 'Electron'),
            # ('seeds_pos_to_remove', 'Remove'),
            ('seeds_pos_avalanche', 'Avalanche'),
            ('seeds_pos_critical', 'Critical'),
            # ('seeds_pos_behind_roi', 'Behind'),
            ('seeds_pos_in_streamer', 'Collided'),
            # ('seeds_pos_to_append', 'New'),
            #
            ('streamer_pos', 'Active'),
            ('streamer_pos_appended', 'New'),
            ('streamer_pos_removed', 'Removed'),
            ]
        self.plot_descs = OrderedDict(
            (k, {'label': v}) for (k, v) in pdl)

        self.seed_keys = [
            'seeds_pos',
            'seeds_pos_ion',
            'seeds_pos_electron',
            'seeds_pos_to_remove',
            'seeds_pos_avalanche',
            'seeds_pos_critical',
            'seeds_pos_behind_roi',
            'seeds_pos_in_streamer',
            'seeds_pos_to_append',
            ]

        self.streamer_keys = [
            'streamer_pos',
            'streamer_pos_removed',
            'streamer_pos_appended',
            ]

        for k, v in self.plot_descs.items():
            if k in self.seed_keys:  # add v after because of label
                self.plot_descs[k] = dict(self.seed_desc, **v)
            elif k in self.streamer_keys:
                self.plot_descs[k] = dict(self.streamer_desc, **v)
            else:
                logger.error('Error: Re-check implementation!')

    def get_xyrz(self, data, key):
        # Get data

        if key in self.seed_keys:
            pos = LoadedData(data=self.data).get_seeds(key=key, idx=self.idx)
        elif key in self.streamer_keys:
            pos = self.dirty_get(self.data, key, self.idx)
        else:
            logger.error('Warning! Invalid key ({})'.format(key))
            return np.aray([]), np.aray([]), np.aray([]), np.aray([])

        x = pos[0, :]
        y = pos[1, :]
        z = pos[2, :]
        r = np.sqrt(pos[0, :]**2 + pos[1, :]**2)
        return x, y, r, z


# create a dict of plotters
plotters = {
    'heads': HeadsPlotter,
    'seeds': SeedsPlotter,
    'seedhead': SeedHeadPlotter,
}

#
