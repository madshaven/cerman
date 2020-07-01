#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Configure plotting. Essentially overwrite mpl style.
'''

# General imports
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler
import logging

# Import from project files
from . import tools

# Settings
FOLDER_PATH = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CermanRCParams():
    ''' Modify matplotlib's rcParams for plotting.
    '''

    # list of options to be evaluated
    # using classmethod since class static variables are not inherited
    @classmethod
    def _get_opt_bool(cls):
        # these keywords may also be used for float or string
        # setting a kw to true here should then use the defaults
        opts = dict()
        opts['clean'] = 'Apply this matplotlib style.'
        opts['cerman'] = 'Apply this matplotlib style.'
        opts['xkcd'] = 'Apply this matplotlib style.'
        opts['default'] = 'Apply this matplotlib style.'
        return opts

    @classmethod
    def _get_opt_float(cls):
        opts = dict()
        return opts

    @classmethod
    def _get_opt_string(cls):
        opts = dict()
        opts['figsizecm'] = 'Use `figsize=w_h` to set size in cm.'
        opts['figsize'] = 'Use `figsize=w_h` to set size in inches, or '
        opts['figsize'] += '`figsize=half/normal/large/huge/column/twocolumn`.'
        opts['dpi'] = 'Set dpi of plot'
        opts['fontsize'] = 'Set fontsize of plot.'
        opts['format'] = 'Set format of saved file.'
        opts['figcolors'] = 'Set number of colors in color cycle.'
        opts['skipcolors'] = 'Skip the first n colors in the cycle.'
        return opts

    @classmethod
    def opt2kwargs(cls, options=None):
        return tools.opt2kwargs(
            options=options,
            bools=cls._get_opt_bool(),
            floats=cls._get_opt_float(),
            strings=cls._get_opt_string(),
            modify=True,  # modifies list, does not warn for unused
            )

    @classmethod
    def opt2help(cls):
        return tools.opt2help(
            bools=cls._get_opt_bool(),
            floats=cls._get_opt_float(),
            strings=cls._get_opt_string(),
            )

    def __init__(self, style='cerman'):
        self.style = style
        self.use(style='default')  # reset to rcDefaults
        self.use()  # set default settings when loaded

    def use(self, style=None):
        ''' Set mpl style. Should be possible to chain styles.'''
        style = self.style if (style is None) else style
        logger.debug(f'Using mpl style: {style}')

        if style == 'default':
            mpl.rcdefaults()
        elif style == 'cerman':
            mpl.style.use(FOLDER_PATH / 'cerman.mplstyle')
        elif style == 'clean':
            # mpl.style.use(FOLDER_PATH / 'cerman.mplstyle')
            mpl.style.use(FOLDER_PATH / 'cerman_clean.mplstyle')
        elif style == 'xkcd':
            # parameters should probably be based on figure size
            plt.xkcd()
            # plt.xkcd(scale=3, length=200, randomness=10)
        else:
            logger.info(f'Could not use style: {style}')

    def use_options(self, options=None):
        if options is None:
            return options
        # todo: add option to add any mpl-style keyword automatically

        # convert option list to dict
        kwargs = self.opt2kwargs(options=options)

        self.use()  # set defaults first
        self._use_options(kwargs)
        return options  # note: trimmed by opt2kwargs

    def color_cycler(self, key=None):
        return color_cycler(key=key)

    def marker_cycler(self, key=None):
        return marker_cycler(key=key)

    @staticmethod
    def change_rcparams(key, value):
        # just a wrapper for nice logging of changes
        logger.debug(f'Set mpl.rc: {key} = {value}')
        mpl.rcParams[key] = value

    def _use_options(self, kwargs):
        '''Set mpl plotting style according to options, return unused options.
        '''
        self._use_options_style(kwargs)
        self._use_options_figsize(kwargs)
        self._use_options_quality(kwargs)
        self._use_options_colors(kwargs)

    def _use_options_style(self, kwargs):
        # iterate here to preserve order of using
        for k, v in kwargs.items():     # assume list
            if (v is True) and k in ['clean', 'cerman', 'xkcd', 'default']:
                self.use(k)

    def _use_options_figsize(self, kwargs):
        if kwargs.get('figsize') is None:
            pass
        elif kwargs.get('figsize') == 'half':
            self.change_rcparams('figure.figsize', (5, 6))
        elif kwargs.get('figsize') == 'large':
            self.change_rcparams('figure.figsize', (16, 10))
        elif kwargs.get('figsize') == 'large':
            self.change_rcparams('figure.figsize', (16, 10))
        elif kwargs.get('figsize') == 'huge':
            self.change_rcparams('figure.figsize', (32, 20))
        elif kwargs.get('figsize') == 'column':
            self.change_rcparams('figure.figsize', (3.5, 2.5))
        elif kwargs.get('figsize') == 'normal':
            self.change_rcparams('figure.figsize', (5.5, 3.9))
        elif kwargs.get('figsize') == 'twocolumn':
            self.change_rcparams('figure.figsize', (7.5, 5.3))
        else:
            # assume 'figsize=float_float'
            w, h = kwargs.get('figsize').split('_')
            size = (float(w), float(h))
            self.change_rcparams('figure.figsize', size)

        if kwargs.get('figsizecm'):
            w, h = kwargs.get('figsizecm').split('_')
            size = (float(w) / 2.54, float(h) / 2.54)
            self.change_rcparams('figure.figsize', size)

    def _use_options_quality(self, kwargs):
        if kwargs.get('dpi'):
            self.change_rcparams('savefig.dpi', int(kwargs.get('dpi')))
        if kwargs.get('fontsize'):
            self.change_rcparams('font.size', float(kwargs.get('fontsize')))
        if kwargs.get('format'):
            self.change_rcparams('savefig.format', str(kwargs.get('format')))

    def _use_options_colors(self, kwargs):
        if kwargs.get('figcolors'):
            key = kwargs.get('figcolors')
            self.change_rcparams('axes.prop_cycle', color_cycler(key))
        if kwargs.get('skipcolors'):
            # assume 'skipcolor=sint'
            # skip the first n colors of colormap
            # invoke this after changing colors
            no = int(kwargs.get('skipcolors'))
            colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']
            cc = cycler(color=colors[no:] + colors[:no])
            self.change_rcparams('axes.prop_cycle', cc)


_color_list_12 = [
    '#1F78B4',   # blue
    '#E31A1C',   # red
    '#33A02C',   # green
    '#FF7F00',   # orange
    '#6A3d9A',   # purple
    '#A6CEE3',   # light blue
    '#FB9A99',   # light red/pink
    '#B2dF8A',   # light green
    '#FdBF6F',   # light orange
    '#CAB2d6',   # light purple
    '#FFFF99',   # yellow
    '#B15928',   # brown
    ]

_color_list_7 = [
    '#E24A33',
    '#348ABD',
    '#988ED5',
    '#777777',
    '#FBC15E',
    '#8EBA42',
    '#FFB5B8',
    ]

_color_list_5 = [  # used for 'clean'
    '#377EB8',
    '#E41A1C',
    '#000000',
    '#984ea3',
    '#ff7f00',
    ]

colors_dict = {
    None: mpl.rcParams['axes.prop_cycle'],
    'mpl': mpl.rcParams['axes.prop_cycle'],
    '5': cycler(color=_color_list_5),
    '7': cycler(color=_color_list_7),
    '12': cycler(color=_color_list_12),
    }

markers_dict = {
    None: cycler(marker=['x', 'd', '*', 'o', 's']),
    '5': cycler(marker=['x', 'd', '*', 'o', 's']),
    }

# .: point
# ,: pixel
# o: circle
# v: triangle_down
# ^: triangle_up
# <: triangle_left
# >: triangle_right
# 1: tri_down
# 2: tri_up
# 3: tri_left
# 4: tri_right
# 8: octagon
# s: square
# p: pentagon
# P: plus (filled)
# *: star
# h: hexagon1
# H: hexagon2
# +: plus
# x: x
# X: x (filled)
# D: diamond
# d: thin_diamond


def color_cycler(key=None):
    ''' Return color cycle for given key. Defaults to rcParams. '''
    if key not in colors_dict:
        msg = f'Possible figcolors are: {list(colors_dict)}'
        logger.warning(f'Wrong figcolor: {key}.' + msg)
        key = None
    return colors_dict[key]


def marker_cycler(key=None):
    ''' Return marker cycle for given key. '''
    if key not in markers_dict:
        msg = f'Possible markers are: {list(markers_dict)}'
        logger.warning(f'Wrong marker: {key}.' + msg)
        key = None
    # note : https://matplotlib.org/3.1.0/api/markers_api.html
    return markers_dict[key]


# # set default settings when this module is loaded
cerman_rc_params = CermanRCParams()


#
