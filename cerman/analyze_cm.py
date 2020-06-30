#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Overall functions for creating plots, including loading of data.
'''

from timeit import default_timer
import logging

from . import analyze
from .analyze.load_data import LoadedData
from .analyze import tools
from .analyze import iteration
from .analyze import simulation
from .analyze import combination
from .analyze.cerman_rc_params import cerman_rc_params
from .analyze.combination.results_data import ResultsData

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def plot_iteration(fpath, plot_kind, glob=None, idxes=None, options=None):
    '''Wrapper for plotting functions for iteration.

    Load data and create all plots for the given file,
    or files in the given directory by globbing.

    Parameters
    ----------
    fpath :     str
                file or folder path
    plot_kind : str
                various
    glob :      str
                globbing pattern for recursive search in given folder
    idxes :     lst[int]
                list of indecies to plot
    options :   lst[str]
                options passed on to plotter
    '''

    if options is None:
        options = []
    options = cerman_rc_params.use_options(options)

    plotters = {}

    # add scatter plotters
    for key, plotter in iteration.scatter.plotters.items():
        plotters[key] = plotter

    longest = max(len(k) for k in plotters)
    doc_strings = {key: plotters[key].__doc__ for key in plotters}
    info_string = '\n'.join('{k:>{l_}} : {v}'.format(k=k, v=v, l_=longest)
                            for k, v in sorted(doc_strings.items()))

    # set the correct plotter to be used
    if (plot_kind not in plotters):
        if (plot_kind not in ['help', 'info']):
            logger.warning(f'Plot kind `{plot_kind}` is not available')
            logger.info('')
        logger.info('Available kinds of plots are:')
        logger.info(info_string)
        logger.info('')
        logger.info('Choose a plot kind and use `help` to see usable options')
        return

    Plotter = plotters[plot_kind]
    if ('help' in options) or ('info' in options):
        Plotter.opt2help()
        return

    # define how to load data and add to plot, for each file
    kwargs = Plotter.opt2kwargs(options)
    def func(fpath):
        logger.debug(f'Loading data, {fpath}')
        loaded_data = LoadedData(fpath)   # load data
        logger.debug(f'Plotting data, {fpath}')
        nos = loaded_data.get_nos(nos=None, idxes=idxes)
        length = len(nos)
        t_start = default_timer()

        # Create plots
        for i, no in enumerate(sorted(nos)):
            t_used = (default_timer() - t_start)
            t_left = t_used * (length / (i + 1) - 1)
            t_left = t_used * (length - i - 1) / (i + 1)
            msg = 'Plot {} of {}: no {}, used {:2.1f} s, remaining {:2.1f} s.'
            msg = msg.format(i + 1, length, no, t_used, t_left)
            logger.info(msg)

            # create plot of iteration
            plotter = Plotter(**kwargs)
            plotter.add_data(data=loaded_data.data, no=no)
            plotter.save_plot()

        # clear memory
        loaded_data = {}

    fpaths = tools.glob_fpath(fpath, glob=glob, post='.pkl')
    if not fpaths:
        logger.info('No files to plot.')
        return

    logger.info(f'Plot kind: {plot_kind}')
    tools.map_fpaths(func=func, fpaths=fpaths, info='Load and plot: {fpath}')


def plot_simulation(fpath, plot_kind, glob=None, options=None):
    '''Wrapper for plotting functions working on one or more simulations.

    Load data and create all plots for the given file,
    or files in the given directory by globbing.

    Parameters
    ----------
    fpath :     str
                file or folder path
    plot_kind : str
                several types available
    glob :      str
                globbing pattern for search in given folder
    options :   lst[str]
                options passed on to plotter
    '''

    if options is None:
        options = []

    # set options for plotting (matplotlib), return unused settings
    options = cerman_rc_params.use_options(options)

    # use multi-mode as default, plotting all given simulation files
    multi_mode = True
    if (options is not None) and ('single' in options):
        options.remove('single')
        multi_mode = False

    # initiate a list of possible plotters
    plotters = {}

    # add gap plotters
    for key, plotter in simulation.gap_plotter.plotters.items():
        plotters[key] = plotter

    # add shadow plotters
    for key, plotter in simulation.shadow_plotter.plotters.items():
        plotters[key] = plotter

    # do not add example plotter and json plotter
    # for key, plotter in analyze.plotterXZ.plotters.items():
    #     plotters[key] = plotter

    # compile a docstring for help on available plotters
    longest = max(len(k) for k in plotters)
    doc_strings = {key: plotters[key].__doc__ for key in plotters}
    info_string = '\n'.join('{k:>{l_}} : {v}'.format(k=k, v=v, l_=longest)
                            for k, v in sorted(doc_strings.items()))

    # set the correct plotter to be used
    if (plot_kind not in plotters):
        if (plot_kind not in ['help', 'info', None]):
            logger.warning(f'Plot kind `{plot_kind}` is not available')
            logger.info('')
        logger.info('Available kinds of plots are:')
        logger.info(info_string)
        logger.info('')
        logger.info('Choose a plot kind and use `help` to see usable options')
        return

    Plotter = plotters[plot_kind]
    if ('help' in options) or ('info' in options):
        Plotter.opt2help()
        return

    # define how to load data and add to plot
    if multi_mode:  # define plotter before func, save later
        kwargs = Plotter.opt2kwargs(options)
        plotter = Plotter(**kwargs)
        def func(fpath):
            logger.debug(f'Loading data, {fpath}')
            loaded_data = LoadedData(fpath)   # load data
            logger.debug(f'Plotting data, {fpath}')
            plotter.add_data(loaded_data.data)      # perform function
            loaded_data = {}                       # clear memory
    else:  # define plotter in func, save in func
        kwargs = Plotter.opt2kwargs(options)
        def func(fpath):
            logger.debug(f'Loading data, {fpath}')
            loaded_data = LoadedData(fpath)   # load data
            logger.debug(f'Plotting data, {fpath}')
            plotter = Plotter(**kwargs)
            plotter.add_data(loaded_data.data)      # perform function
            plotter.save_plot()
            loaded_data = {}                       # clear memory

    # find files to plot
    fpaths = tools.glob_fpath(fpath, glob=glob, post='.pkl')
    if not fpaths:
        logger.info('No files to plot.')
        return

    logger.info(f'Plot kind: {plot_kind}')
    tools.map_fpaths(func=func, fpaths=fpaths, info='Load and plot: {fpath}')
    if multi_mode:
        plotter.save_plot()


def plot_results(fpath, plot_kind=None, glob=None, options=None):
    '''Wrapper for plotting functions.

    Load data and create all plots for the given file,
    or files in the given directory.

    Parameters
    ----------
    fpath :     str
                file or folder path to look for files
    plot_kind : str
                simulation parameter key(s) to be used for x-axis
    glob :      str
                globbing pattern for recursive search in given folder
    options :   list(str)
                options for plotting
    '''

    if options is None:
        options = []

    # set main/general options for plot, return unused options
    options = cerman_rc_params.use_options(options)
    Plotter = combination.results_plotter.ResultsPlotterWrapper

    if ('help' in options) or ('info' in options):
        Plotter.opt2help()
        return

    def to_to_func(fpath):
        kwargs = Plotter.opt2kwargs(options)
        results_data = tools.json_load(fpath)
        plotter = Plotter(x_keys=plot_kind, **kwargs)
        plotter.add_data(results_data)
        plotter.save_plot()
        results_data = {}

    fpaths = tools.glob_fpath(fpath, glob=glob, post='.stat')
    tools.map_fpaths(func=to_to_func, fpaths=fpaths, info=None)


def plot_json(fpath, glob=None, options=None):
    '''Wrapper for plotting functions.

    Load data and create all plots for the given file,
    or files in the given directory by globbing.

    Parameters
    ----------
    fpath :     str
                file or folder path
    glob :      str
                globbing pattern for recursive search in given folder
    '''

    plot_kind = 'json'  # just to keep more similar workflow as above

    if options is None:
        options = []
    options = cerman_rc_params.use_options(options)

    Plotter = analyze.plotterXZ.plotters[plot_kind]

    if ('help' in options) or ('info' in options):
        Plotter.opt2help()
        return

    def func(fpath):
        kwargs = Plotter.opt2kwargs(options)
        plotter = Plotter(**kwargs)
        plotter.add_data(fpath)
        plotter.save_plot()

    fpaths = tools.glob_fpath(fpath, glob=glob, post='.json')
    if not fpaths:
        logger.info('No files to plot.')
        return

    logger.info(f'Plot kind: {plot_kind}')
    tools.map_fpaths(func=func, fpaths=fpaths, info='Load and plot: {fpath}')


def plot_parameters(fpath, glob=None, options=None):
    '''Create a matrix plot of input parameters.

    Load data and create all plots for the given file,
    or files in the given directory.

    Parameters
    ----------
    fpath :     str
                file or folder path to look for files
    glob :      str
                globbing pattern for recursive search in given folder
    options :   list(str)
                options for plotting (type `help` for details)
    '''

    if options is None:
        options = []
    # set matplotlib options
    options = cerman_rc_params.use_options(options)

    # choose plotter
    Plotter = combination.matrix_plotter.ParameterPlotter
    if ('help' in options) or ('info' in options):
        Plotter.opt2help()
        return

    # get paths to load data from
    logger.info('Finding parameter files')
    fpaths = tools.glob_fpath(fpath, glob=glob)
    if not fpaths:
        logger.info('No files to plot.')
        return

    # define how to load data and add to plot
    kwargs = Plotter.opt2kwargs(options)
    plotter = Plotter(**kwargs)

    def func(fpath):
        logger.debug(f'Loading data: {fpath}')
        plotter.add_fpath(fpath)

    tools.map_fpaths(func=func, fpaths=fpaths, info=None)
    plotter.save_plot()


def plot_reason_for_stop(fpath, glob=None, options=None):
    '''Create a matrix plot of stop reasons.

    Load data and create all plots for the given file,
    or files in the given directory.

    Parameters
    ----------
    fpath :     str
                file or folder path to look for files
    glob :      str
                globbing pattern for recursive search in given folder
    options :   list(str)
                options for plotting (type `help` for details)
    '''

    if options is None:
        options = []
    # set matplotlib options
    options = cerman_rc_params.use_options(options)

    # choose plotter
    Plotter = combination.matrix_plotter.StopReasonPlotter
    if ('help' in options) or ('info' in options):
        Plotter.opt2help()
        return

    # get paths to load data from
    logger.info('Finding log files')
    fpaths = tools.glob_fpath(fpath, glob=glob, post='.log')
    if not fpaths:
        logger.info('No files to plot.')
        return

    # define how to load data and add to plot
    kwargs = Plotter.opt2kwargs(options)
    plotter = Plotter(**kwargs)

    def func(fpath):
        logger.debug(f'Loading data: {fpath}')
        plotter.add_fpath(fpath)

    tools.map_fpaths(func=func, fpaths=fpaths, info=None)
    plotter.save_plot()


def create_archive(fpath=None, glob=None, options=None):
    ''' Create and/or modify an archive of parsed simulation results.

    Parameters
    ----------
    fpath : str
            file or folder path to look for files
    glob  : str
            globbing pattern
    options :   list(str)
                options for plotting (type `help` for details)
    '''
    if options is None:
        options = []

    if ('help' in options) or ('info' in options):
        ResultsData.opt2help()
        return

    # note: fpath=None implies to infer fpath from first added file
    kwargs = ResultsData.opt2kwargs(options=options)
    results_data = ResultsData(**kwargs)

    # glob for a list of files
    fpaths = tools.glob_fpath(fpath, glob=glob, post='.pkl')

    def to_do_func(fpath):
        results_data.modify(fpath=fpath)

    logger.info('Creating/modifying statistics')
    tools.map_fpaths(func=to_do_func, fpaths=fpaths, info=None)


#
