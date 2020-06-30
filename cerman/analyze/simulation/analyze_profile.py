#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Everything needed to make sense of profiling data.
'''

# Imports for data manipulation
import pstats
import numpy as np
from pathlib import Path
import logging

# Imports for plotting
from matplotlib.backends.backend_agg import FigureCanvasAgg as Canvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib import ticker
from datetime import timedelta

# Import from project files
from .. import tools
from ..cerman_rc_params import cerman_rc_params

# Settings
PROFILE_SUFFIX = '.profile'
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AnalyseProfile():
    ''' A class to parse `options` and analyze profile. '''

    # note: this class is perhaps a bit too much
    #       it was made as an attempt
    #       for something that can be streamlined later.

    _opt_bool = {
        }
    _opt_string = {
        'mode' : "Display ('display'), "
                 "plot ('[ttct]', 'tt', 'ct'), or save ('text')",
        }
    _opt_float = {
        'no' : 'Number of data to include'
        }

    @classmethod
    def opt2kwargs(cls, options=None):
        return tools.opt2kwargs(
            options=options,
            bools=cls._opt_bool,
            floats=cls._opt_float,
            strings=cls._opt_string,
            )

    @classmethod
    def opt2help(cls):
        return tools.opt2help(
            bools=cls._opt_bool,
            floats=cls._opt_float,
            strings=cls._opt_string,
            )

    def __init__(self, fpath, glob=None, options=None):
        if options is None:
            options = []

        # settings for plotting
        options = cerman_rc_params.use_options(options)

        # use multi-mode as default, plotting all given simulation files
        self.multi_mode = True
        if (options is not None) and ('single' in options):
            options.remove('single')
            self.multi_mode = False

        # help for options
        if ('help' in options) or ('info' in options):
            self.opt2help()
            return

        # find files
        self.fpaths = tools.glob_fpath(fpath, glob=glob, post=PROFILE_SUFFIX)
        if not self.fpaths:
            logger.info('No files found.')
            return

        # set kwargs for function
        self.kwargs = self.opt2kwargs(options)
        self.data = None

        info = 'Loading: {fpath}'
        tools.map_fpaths(func=self.to_do_func, fpaths=self.fpaths, info=info)
        self._exit()

    def to_do_func(self, fpath):
        ''' Perform action on single file. '''
        if self.multi_mode:
            self._to_do_multi(fpath)
        else:
            self._to_do_single(fpath)

    def _to_do_single(self, fpath):
        ''' What to do with single data. '''
        self.data = load_profile_data(str(fpath))
        self._mode_function()

    def _to_do_multi(self, fpath):
        ''' What to do with multi data. '''
        data = load_profile_data(str(fpath))
        if self.data:
            self.data = combine_data(self.data, data)
        else:
            self.data = data
        # do mode_function at end

    def _mode_function(self):
        ''' How to present data. Plot / display / file. '''
        no = self.kwargs.get('no')
        no = None if (no is None) else int(no)  # just force an int if needed

        mode = self.kwargs.get('mode')
        if mode is None or mode == 'plot':
            mode = 'ttct'
        if mode in ['d', 'display']:
                display_data(self.data, no=no)
        elif mode in ['ttct', 'ct', 'tt']:
                plot_data(self.data, plot_kind=mode, no=no)
        elif mode in ['f', 'file', 'txt', 'text']:
                save2txt(self.data, no=no)
        else:
            logger.info(f'Unknown mode: {self.mode}')
            return

    def _exit(self):
        ''' Perform clean-up, if needed. '''
        if self.multi_mode:
            self._mode_function()
        else:
            pass


def load_profile_data(fpath):
    ''' Load data from a profile file and return a dictionary.

    Parameters
    ----------
    fpath : str or Path
            file or folder path

    Returns
    -------
    dict
            data
    '''

    # Check for correct input
    fpath = Path(fpath)
    if not fpath.is_file():
        msg = 'The file "{}" does not exist.'.format(fpath)
        raise FileNotFoundError(msg)
    if not fpath.suffix == PROFILE_SUFFIX:
        msg = 'The file "{}" is not a "{}"-file.'.format(fpath, PROFILE_SUFFIX)
        raise FileNotFoundError(msg)

    # Load input to stats-object
    stats = pstats.Stats(str(fpath))

    # Create data structure for output
    keys = ['cc', 'nc', 'tt', 'ct', 'f', 'f0', 'f1', 'f2', 'f3']
    data = {k: [] for k in keys}

    # Copy data to output dictionary
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        data['cc'].append(cc)
        data['nc'].append(nc)
        data['tt'].append(tt)
        data['ct'].append(ct)
        data['f0'].append(func[0])  # path
        data['f1'].append(func[1])  # line no
        data['f2'].append(func[2])  # function
        data['f3'].append(Path(func[0]).name)
        f = '{}:{}({})'.format(Path(func[0]).name, func[1], func[2])
        data['f'].append(f)

    # Convert data lists to np arrays
    for key in data:
        data[key] = np.array(data[key])

    # Add file information to data
    data['fpath'] = fpath

    # Add sorted indecies to data
    data['tt_srt_idx'] = np.argsort(data['tt'])
    data['ct_srt_idx'] = np.argsort(data['ct'])

    return data


def combine_data(d1, d2):
    ''' Return combined data from both dicts. '''
    # assume that if d['f'] are the same, that they are the same

    # rename, preserving fpath
    dout = {key: np.array(d1[key]).tolist() for key in d1}
    d2 = {key: np.array(d2[key]).tolist() for key in d2}

    for f in d2['f']:
        # print(f)
        d2i = d2['f'].index(f)
        if f in dout['f']:
            doi = dout['f'].index(f)
            for key in ['cc', 'nc', 'tt', 'ct']:
                dout[key][doi] += d2[key][d2i]
        else:
            for key in dout:
                if not key == 'fpath':
                    dout[key].append(d2[key][d2i])

    for key in dout:
        dout[key] = np.array(dout[key])

    dout = {key: np.array(dout[key]) for key in dout}

    # Keep fpath from first
    dout['fpath'] = d1['fpath']

    # Add sorted indecies to data
    dout['tt_srt_idx'] = np.argsort(dout['tt'])
    dout['ct_srt_idx'] = np.argsort(dout['ct'])

    return dout


def data2str(data, no=None):
    ''' Convert profile data to string objects.

    Parameters
    ----------
    data  : dict
            profile data
    no    : int
            number of data to include

    Returns
    -------
    tt_str : str
             total time sorted string
    ct_str : str
             cummulative time sorted string
    '''

    if no is None:
        no = 10

    # Create string header
    h1 = ['ncalls', 'tottime', 'percall',
          'cumtime', 'percall', 'fname:line(func)']
    h2 = ['', '[s]', '[ms]', '[s]', '[ms]', '']
    h3 = ['-' * len(i) for i in h1]
    fmt = ' {: >8}  {: >8}  {: >8}  {: >8}  {: >8}  {}'
    header = [fmt.format(*h) for h in [h1, h2, h3]]

    # Define data to string line function
    def get_str_by_idx(j):
        nc = data['nc'][j]
        tt = data['tt'][j]
        ct = data['ct'][j]
        f = data['f'][j]
        info = [nc, tt, tt / nc * 1e3, ct, ct / nc * 1e3, f]
        fmt = ' {: >8.0f}  {: >8.3f}  {: >8.3f}  {: >8.3f}  {: >8.3f}  {}'
        return fmt.format(*info)

    # Find the indecies for the sorted data
    tt_idx = [data['tt_srt_idx'][-i - 1] for i in range(no)]
    ct_idx = [data['ct_srt_idx'][-i - 1] for i in range(no)]

    # Initiate strings
    tt_str  = '\n'.join(header) + '\n'
    ct_str  = '\n'.join(header) + '\n'
    tt_str += '\n'.join(get_str_by_idx(idx) for idx in tt_idx)
    ct_str += '\n'.join(get_str_by_idx(idx) for idx in ct_idx)

    return tt_str, ct_str


def display_data(data, no=None):
    tt_str, ct_str = data2str(data, no=no)
    logger.info('')
    logger.info('Total time')
    logger.info(f'{tt_str}')
    logger.info('')
    logger.info('Cummulative time')
    logger.info(f'{ct_str}')
    logger.info('')


def plot_data(data, plot_kind=None, no=None, fpath=None):
    ''' Create a plot of profile data.

    Parameters
    ----------
    data  : dict
            profile data
    plot_kind : str
            ttct, tt, ct
    no    : int
            number of data to include
    fpath : str
            plot file path
    '''

    if plot_kind is None:
        plot_kind = 'ttct'

    if plot_kind not in ['ttct', 'ct', 'tt']:
        logger.warning('Invalid plot type chosen. "{}"'.format(plot_kind))
        logger.warning('Changing to `ttct`.')
        plot_kind = 'ttct'

    if no is None:
        if plot_kind == 'ttct':
            no = 6
        else:
            no = 10

    # Get 'fpath' if not specified
    if fpath is None:
        fpath = tools.get_fpath(data, 'profile_' + plot_kind)

    # Find total time and correct indecies
    totaltime = np.array(data['tt']).sum()
    idx_tt = data['tt_srt_idx'][-no:]
    idx_ct = data['ct_srt_idx'][-no:]
    index = np.arange(no)
    bar_width = 0.35

    # Data for tt plot
    yl_tt = data['f2'][idx_tt]                      # labels
    yl_tt = data['f'][idx_tt]                      # labels
    pt_tt = data['tt'][idx_tt] / totaltime * 100    # percent
    pc_tt = data['ct'][idx_tt] / totaltime * 100    # percent
    xt_tt = data['tt'][idx_tt]                      # value total time
    xc_tt = data['ct'][idx_tt]                      # value cum time
    fn_tt = data['nc'][idx_tt] / data['nc'][idx_tt[-1]]  # call no fraction

    # Data for ct plot
    yl_ct = data['f2'][idx_ct]                      # labels
    yl_ct = data['f'][idx_ct]                      # labels
    pt_ct = data['tt'][idx_ct] / totaltime * 100    # percent
    pc_ct = data['ct'][idx_ct] / totaltime * 100    # percent
    xt_ct = data['tt'][idx_ct]                      # value total time
    xc_ct = data['ct'][idx_ct]                      # value cum time
    fn_ct = data['nc'][idx_ct] / data['nc'][idx_ct[-1]]  # call no fraction

    # Create figure and axes
    if plot_kind == 'ttct':
        fig = Figure()  # create new figure
        gs = GridSpec(2, 1, figure=fig)
        ax0 = fig.add_subplot(gs[0])   # add axis
        ax1 = fig.add_subplot(gs[1], sharex=ax0)   # add axis
    else:
        fig = Figure()  # create new figure
        gs = GridSpec(1, 1, figure=fig)
        ax = fig.add_subplot(gs[0])   # add axis

    # Define plot function
    def plot_to_ax(ax, index, yl, xt, xc, bar_width):
        bt = ax.barh(index, xt, height=bar_width,
                     label='total time', color='r', alpha=0.5)
        bc = ax.barh(index + bar_width, xc, height=bar_width,
                     label='cumulative time', color='b', alpha=0.4)
        ax.set_yticks(index + bar_width)
        ax.set_yticklabels(yl)
        formatter = ticker.FuncFormatter(lambda x, pos: timedelta(seconds=x))
        ax.xaxis.set_major_formatter(formatter)
        return bt, bc

    # Define annotation function
    def annotate_ax(ax, plot, xp, nc):
        for i, patch in enumerate(plot.get_children()):
            bl = patch.get_xy()
            x = 0.0 + bl[0]
            y = 0.4 * patch.get_height() + bl[1]
            # Whitespace is an easy way of adding some padding.
            string = '  t/T: {: >5.2f} %, n/n0: {:1.3f}'.format(xp[i], nc[i])
            ax.annotate(string, xy=(x, y), ha='left', va='center')

    # Plot two plots
    if plot_kind == 'ttct':

        leg_ax = ax1
        bt_tt, bc_tt = plot_to_ax(ax0, index, yl_tt, xt_tt, xc_tt, bar_width)
        bt_ct, bc_ct = plot_to_ax(ax1, index, yl_ct, xt_ct, xc_ct, bar_width)

        td = timedelta(seconds=int(totaltime))
        ax0.set_title(f'Total time: {td}')
        # ax0.get_xaxis().set_ticklabels([])  # no x-tick labels
        annotate_ax(ax0, bt_tt, pt_tt, fn_tt)
        # annotate_ax(ax0, bc_tt, pc_tt, fn_tt)

        # annotate_ax(ax1, bt_ct, pt_ct, fn_ct)
        annotate_ax(ax1, bc_ct, pc_ct, fn_ct)

    # Plot total time data
    elif plot_kind == 'tt':
        ax0 = ax
        leg_ax = ax
        bt_tt, bc_tt = plot_to_ax(ax0, index, yl_tt, xt_tt, xc_tt, bar_width)
        annotate_ax(ax0, bt_tt, pt_tt, fn_tt)
        annotate_ax(ax0, bc_tt, pc_tt, fn_tt)
        td = timedelta(seconds=int(totaltime))
        ax.set_title(f'Sorted by total time')

    # Plot cumulative time data
    elif plot_kind == 'ct':
        ax1 = ax
        leg_ax = ax
        bt_ct, bc_ct = plot_to_ax(ax1, index, yl_ct, xt_ct, xc_ct, bar_width)
        annotate_ax(ax1, bt_ct, pt_ct, fn_ct)
        annotate_ax(ax1, bc_ct, pc_ct, fn_ct)
        td = timedelta(seconds=int(totaltime))
        ax.set_title(f'Sorted by cumulative time')

    else:
        logger.error('Something went wrong while plotting profile!')

    leg_ax.set_xlabel('Time spent')
    leg_ax.legend(loc=4)

    # Save plot
    logger.info(f'Saving plot: {fpath}')
    Canvas(fig).print_figure(str(fpath))
    fig.clf()


def save2txt(data, no=None, fpath=None):
    ''' Save profiling data to text file.

    The functions are sorted by total time and cummulative time.
    The first 'no' data are used.

    Parameters
    ----------
    data  : dict
            profile data
    no    : int
            number of data to include
    fpath : str
            save file path
    '''

    if no is None:
        no = 20

    if fpath is None:
        fpath = tools.get_fpath(data, 'profile', ext='txt')

    string = ''
    tt_s, ct_s = data2str(data, no)

    msg = 'Total {} calls in {} seconds.'
    string += msg.format(data['nc'].sum(), data['tt'].sum())
    string += '\n\n'

    string += 'Sorted by total time:\n'
    string += tt_s
    string += '\n'
    string += 'Sorted by cummulative time:\n'
    string += ct_s

    logger.info(f'Writing file: {fpath}')
    with open(fpath, 'w') as f:
        f.write(string)

#
