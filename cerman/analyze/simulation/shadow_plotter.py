#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Functions for plotting streamer shadow.
'''

# idea: Add averaging line to the shadow
# idea: Add 3D shadow
# idea: Add detailed shadow (details/scatter, existing+new)

import numpy as np
import logging

from .. import plotterXYRZ

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE PLOTTER                                                #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ShadowPlotter(plotterXYRZ.PlotterXYRZ):
    '''Plot streamer shadow.'''

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Initialize                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def __init__(self, title=False, fpath_head='shadow',
                 xlabel=r'Position, $x$ [mm]',
                 ylabel=r'Position, $y$ [mm]',
                 rlabel=r'Position, $r$ [mm]',
                 zlabel=r'Position, $z$ [mm]',
                 scale=1e3, axes_visible='xy', xz_ratio=False,
                 xscale=None, yscale=None,
                 rscale=None, zscale=None,
                 scatter=True,
                 **kwargs):

        super().__init__(**kwargs)

        self.title = 'Shadow' if title is True else title
        self.fpath_head = 'shadow' if fpath_head is True else fpath_head
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.rlabel = rlabel
        self.zlabel = zlabel

        self.xscale = xscale or scale
        self.yscale = yscale or scale
        self.rscale = rscale or scale
        self.zscale = zscale or scale

        # plot setup
        # self.xz_ratio = 2  # for xz&yz
        self.xz_ratio = xz_ratio
        self.axes_visible = axes_visible

        self.symx = False  # symmetric x-axis
        self._zmin = 0    # explicitly set this one

        self.scatter_on = scatter
        self.scatter_desc = {'s': 3, 'marker': 'o', 'alpha': 0.3,
                             'label': None}

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Manage data                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def get_xyrzt(self, data):
        # Get data
        pos, no, key = get_shadow(data)
        logger.debug('Shadow key: {}'.format(key))

        # get time
        ts = np.array(data['sim_time']).reshape(-1)
        nos = np.array(data['no']).reshape(-1)
        idx = np.array([np.argwhere(nos == n)[0][0] for n in no])
        t = np.array([ts[i] for i in idx])

        # arrange return values
        x = pos[0, :]
        y = pos[1, :]
        z = pos[2, :]
        r = np.sqrt(pos[0, :]**2 + pos[1, :]**2)
        t = t
        return x, y, r, z, t


# store a dict of available plotters
plotters = {
    'shadow': ShadowPlotter,
}


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       GET SHADOW                                                    #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_shadow(data, key=None, idx=None, no=None):
    '''Analyze data to get shadow.

    Parameters
    ----------
    data  : dict
            loaded data
    key   : str
            preferred shadow key
    idx   : int
            data idx to get, defaults to get all
    no    : int
            data iteration no to plot,
            overrides 'idx' if specified

    Returns
    -------
    pos
        shadow positions
    no
        iteration number of shadow positions
    key
        key used to obtain shadow
    '''
    # idea: add option to strip repeated values
    # idea: add some more advanced analyser to find branching
    # todo: add func to get shadow from heads_dict

    # Override 'idx' if 'no' is specified
    if no is not None:
        idx = data['no'].index(no)
    if idx == -1:
        idx = len(data['no']) - 1

    # Possible ways of plotting a streamer, in preferred order
    key_lst = [
        'streamer_pos',                 # all current positions
        'streamer_pos_appended',        # positions added this iteration
        'seeds_pos_critical',           # critical avalanches this iteration
        'streamer_pos_appended_all',    # added, but maybe removed this iter
        'streamer_pos_removed',         # removed this iter
        'streamer_pos_removed_all',     # removed this iter, all
        ]

    # Choose the best key
    if key not in data:
        for k in reversed(key_lst):
            if k in data:
                key = k

    # Return nothing if no data was found
    if key is None:
        msg = 'Not possible to get streamer from {}.'
        logger.warning(msg.format(data['header']['fpath']))
        return np.zeros((3, 0)), np.zeros(0), key

    # Find non-empty indecies
    l = len(data[key])
    idxes = [i for i in range(l) if (data[key][i].size > 0)]
    # Return data only for one 'no' or 'idx', if given
    # If 'idx' is None it is not in 'idxes'
    if idx in idxes:
        idxes = [idx]

    if not idxes:
        msg = 'No streamer points in {}.'
        logger.info(msg.format(data['header']['fpath']))

    # Combine data
    pos = np.hstack([np.zeros((3, 0))] +
                    [data[key][i].reshape(3, -1) for i in idxes])
    no = np.array([data['no'][i]
                  for i in idxes
                  for j in range(data[key][i].shape[1])]
                  )

    return pos, no, key

#
