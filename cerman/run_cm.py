#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Script for control command line.

This program parses command line arguments.
The parsed arguments are passed on to the appropriate function.
The function interprets the arguments
and invokes deeper level functions
(having more explicit parameters).
'''

# general imports
from pathlib import Path
import argparse
import logging
import datetime
import subprocess
import sys
from sys import version_info
import shutil

# note: relevant parts of CerMan is imported below when needed

# note: f-strings will anyway give error for python < 3.6
if not ((version_info.major == 3) and (version_info.minor >= 6)):
    msg = 'Python3.6 or above required. Python{}.{} is not supported.'
    msg = msg.format(version_info.major, version_info.minor)
    raise SystemExit(msg)

# hack: set as global since it is an easy solution, consider a fix later
# note: these are modified by the `entry_point` function
global pargs
global fpath


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE INPUT PARSER                                           #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

name_string = r'''
    _____          __  __
   / ____|        |  \/  |
  | |     ___ _ __| \  / | __ _ _ __
  | |    / _ \ '__| |\/| |/ _` | '_ \
  | |___|  __/ |  | |  | | (_| | | | |
   \_____\___|_|  |_|  |_|\__,_|_| |_|

'''

# define actions to add to argument parser
parser_arg_list = [
    (('action', ), {
    'type': str.lower,
    'help': 'Main action to perform. Use "help" for available actions.'
    }),
    (('kind', ), {
    'type': str.lower, 'nargs': '?', 'default': None,
    'help': 'Kind of action (optional), needed for e.g. plot kind/type.'
    }),
    (('-f', '--fpath', ), {
    'type': str, 'default': None,
    'help': 'Path to folder or file to use. Defaults to current folder.'
    }),
    (('-g', '--glob', ), {
    'type': str, 'metavar': 'GLOB', 'default': None,
    'help': 'Globbing pattern used for searching for files.'
    }),
    (('-r', '--range', ), {
    'type': int, 'nargs': 2, 'metavar': ('START', 'END'),
    'help': 'Range, e.g. iteration number range.'
    }),
    (('-o', '--options', ), {
    'type': str, 'metavar': 'OPT',
    'help': 'Options, separated by space, e.g. "opt1 opt2=off opt3=str1_str2".'
    }),
    (('-n', '--number', ), {
    'type': int, 'metavar': 'NO',
    'help': 'Number, e.g. number of items to display/plot.'
    }),
    (('--nice', ), {
    'type': int, 'metavar': 'NICE', 'default': 0,
    'help': 'Control how `nice` a sub-process is.'
    }),
    (('-d', '--debug', ), {
    'action': 'count', 'default': 0,
    'help': 'Display debug information. '
         'Invoke twice for verbose debug.'
    }),
    (('-q', '--quiet', ), {
    'action': 'count', 'default': 0,
    'help': 'Display warnings only. '
         'Invoke twice for errors only.'
    }),
    (('-l', '--logging', ), {
    'action': 'count', 'default': 0,
    'help': f'Create log file from {__name__}.'
    }),
    (('-m', '--mp', ), {
    'type': int, 'default': 0, 'metavar': 'MP',
    'help': 'Number of threads to use, invoking multiprocessing.'
    }),
]


# defining this here instead of making it explicit global
parser = argparse.ArgumentParser(
    description=name_string,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def parse_arguments(args=None):
    # return parsed arguments

    # use commandline arguments as default
    if args is None:
        args = sys.argv[1:]

    # add arguments to the parser
    for (_a, _kwa) in parser_arg_list:
        parser.add_argument(*_a, **_kwa)

    # parse the arguments
    pargs = parser.parse_args(args)

    if pargs.options is not None:
        pargs.options = [opt.strip(',') for opt in pargs.options.split()]
    elif pargs.kind in ['help', 'info']:
        # the most likely interpretation of asking for help
        # note: pargs.kind to take precedence over options for help
        pargs.options = ['help']
    if pargs.range is None:
        pargs.idxes = None
    else:
        pargs.idxes = list(range(pargs.range[0], pargs.range[1]))

    return pargs


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       CONFIGURE LOGGING                                             #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# defining this here instead of making it explicit global
logger = logging.getLogger('cerman')


def configure_logger():
    # get logger

    # configure level and format
    log_fmt = logging.Formatter('%(message)s')
    log_lvl = logging.INFO

    if pargs.debug > 0:
        log_lvl = logging.DEBUG
    if pargs.debug > 1:
        log_lvl = 5
        msg = '%(filename)s:%(lineno)d: %(message)s'
        log_fmt = logging.Formatter(msg)
    if pargs.quiet > 0:
        log_lvl = logging.WARNING
    if pargs.quiet > 1:
        log_lvl = logging.ERROR
    logger.setLevel(log_lvl)

    # configure logging to screen
    ch = logging.StreamHandler()
    ch.setFormatter(log_fmt)
    ch.setLevel(logger.getEffectiveLevel())
    logger.addHandler(ch)

    # configure logging to file
    if pargs.logging:
        fh = logging.FileHandler('./cerman.log', mode='w')
        fh.setFormatter(log_fmt)
        fh.setLevel(logger.getEffectiveLevel())
        logger.addHandler(fh)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE ACTIONS                                                #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class Action():

    # keep track of all possible actions
    all_actions = dict()

    def __init__(self, func, keys, opts=None):
        ''' A class to keep track of `actions` that can be performed.

        Parameters
        ----------
        func :  function
                the function that is called when invoking the action
        key :   str
                name of the action
        keys :  lst(str)
                name of action as well as abbreviations
        opts :  lst(str)
                defines which command line arguments the given action can use
        '''

        self.key = keys[0]
        self.keys = keys
        self.opts = opts or ''  # hack when opts is None
        self.func = func
        self.help = self.parse_doc(func.__doc__)

    @classmethod
    def add(cls, func, keys, opts):
        ''' Create a new action and add it to `all_actions`. '''
        self = cls(func, keys, opts)
        self._add_to_all_actions()
        logger.log(5, f'Added {func.__name__} for {keys}')

    @classmethod
    def call(cls, key, *_args, **_kwargs):
        ''' Find the action for the given key and invoke it. '''
        self = None
        for _, action in cls.all_actions.items():
            if key in action.keys:
                self = action

        if self:
            self(*_args, **_kwargs)
        else:
            # print usage if the given action is not valid
            parser.print_usage()
            msg = f'Error! Unrecognized action: {pargs.action}. '
            msg += 'Type `help` for usage.'
            raise SystemExit(msg)

    @staticmethod
    def parse_doc(docstring):
        ''' Strip unneeded whitespace and newlines. '''
        if docstring is None:
            return ''
        lines = [li.strip() for li in docstring.splitlines()]
        lines = [li + ' ' if (li != '') else '\n' for li in lines]
        return ''.join(lines).strip()

    def __call__(self, *_args, **_kwargs):
        self.func(*_args, **_kwargs)

    def _add_to_all_actions(self):
        if not self._validate():
            logger.error('Implementation error.')
            raise SystemExit

        self.all_actions[self.key] = self

    def _validate(self):
        # check that self.keys are not anywhere in all_actions
        if self.key in self.all_actions:
            logger.error(f'Error! {key} is already added.')
            return False

        for sk in self.keys:
            for _, other in self.all_actions.items():
                if sk in other.keys:
                    msg = f'Error! "{sk}" is already in "{other.key}".'
                    logger.error(msg)
                    return False

        return True


class addToActions(object):
    def __init__(self, keys, opts=None):
        ''' Decorator to store a function in Action.all_actions. '''
        self.keys = keys
        self.opts = opts

    def __call__(self, f):
        Action.add(f, self.keys, self.opts)
        return f


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       SIMULATE                                                      #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@addToActions(['simulate', 'sim', ], ['-f', '-g'])
def _action_simulate():
    ''' Simulate file(s) by importing simulate function from cerman.'''
    import cerman.tools
    import cerman.simulate_cm

    logger.info('Simulate within same process')
    fpaths = cerman.tools.glob_fpath(fpath, glob=pargs.glob, post='.json')
    func = cerman.simulate_cm.simulate
    cerman.tools.map_fpaths(fpaths, func, info='Simulating {fpath}',
                           worker_nos=False)


@addToActions(['simulate_sub', 'sims', ], ['-f', '-g'])
def _action_simulate_sub():
    ''' Simulate file(s) via subprocess.
        Enables basic optimization enabled.
    '''
    import cerman.tools
    # get path to simulation program
    _mname = 'simulate_cm.py'
    _folder = Path(__file__).parent
    mpath = Path('')
    try:
        # let's see if it is in path
        cmd = ['which', _mname]
        mpath = subprocess.check_output(cmd).strip().decode('ascii')
        mpath = Path(mpath)
        msg = f'Using "{_mname}" from path.'
        logger.debug(msg)
        _in_path = True
    except subprocess.CalledProcessError:
        # use from cerman.py folder instead
        msg = f'"{_mname}" not in path. Checking {_folder}.'
        logger.debug(msg)
        mpath = _folder / _mname
        _in_path = False

    # validate that simulation program is available
    if not mpath.is_file():
        logger.error(f'Error! Could not find "{mpath}".')
        logger.info(f'Please add "{_mname}" to path or "{mpath.parent}".')
        raise SystemExit

    # let's see if `nice` is available
    try:
        cmd = ['which', 'nice']
        subprocess.check_output(cmd)
    except subprocess.CalledProcessError:
        if not pargs.nice == 0:
            pargs.nice = 0
            msg = 'Warning! Could not locate `nice`. Running without...'
            logger.error(msg)

    logger.info('Simulate in subprocesses')
    fpaths = cerman.tools.glob_fpath(fpath, glob=pargs.glob, post='.json')
    func = _action_simulate_sub_work

    if pargs.mp == 0:
        worker_nos = False
    else:
        worker_nos = pargs.mp

    cerman.tools.map_fpaths(fpaths, func, info='Simulating {fpath}',
                           worker_nos=worker_nos, mpath=mpath, nice=pargs.nice)


def _action_simulate_sub_work(fpath, mpath=None, nice=0):
    ''' Work function used by `_action_simulate_sub`.'''
    # note: this function is defined at top level to be pickle-able

    cmd = []
    if nice != 0:
        # note: nice stacks if applied from elsewhere as well
        cmd += ['nice', '-n', str(nice)]
    # note: test for python >= 3.6 done at top of file
    # note: invoking python is not needed if _in_path, however, -OO...
    cmd += ['python3']      # python interpreter
    cmd += ['-OO']          # disable debugging and assertions
    cmd += [str(mpath)]     # path to program
    cmd += [str(fpath)]     # path to input
    logger.debug(f'Calling subprocess: "{cmd}"')
    subprocess.call(cmd)
    # Note! Calling subprocess like this may be unsafe.
    # However, it is assumed that the user is sane.


@addToActions(['dump_defaults', ], ['-f'])
def _action_dump_defaults():
    ''' Dump default parameters to file.
    '''
    from cerman.simulation_input.input_tools import dump_defaults as func
    logger.info('Dumping default simulation parameters')
    func(fpath=fpath)


@addToActions(['create_input', 'ci', ], ['-f', '-g'])
def _action_create_input():
    ''' Create several simulation files from one single file.
    '''
    import cerman.tools
    from cerman.simulation_input.input_tools import create_input_files as func

    logger.info('Creating input files')
    fpaths = cerman.tools.glob_fpath(fpath, glob=pargs.glob)
    info = 'Expanding input file: {fpath}'
    cerman.tools.map_fpaths(func=func, fpaths=fpaths, info=info)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       ANALYZE                                                       #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@addToActions(['profile', 'pf', ], ['-f', '-g', '-n'])
def _action_profile():
    ''' Analyze profiling data profiler.
    '''
    from cerman.analyze.simulation.analyze_profile import AnalyseProfile

    logger.info('Analyze profile data')
    AnalyseProfile(fpath, glob=pargs.glob, options=pargs.options)


@addToActions(['inspect_data', 'id', ], ['-f', '-g', '-o'])
def _action_inspect_data():
    ''' Print data in save file.
    '''
    from cerman.analyze.inspect_data import inspect_data as func
    logger.info('Printing saved data')
    func(fpath, glob=pargs.glob, idxes=pargs.idxes, options=pargs.options)


@addToActions(['create_archive', 'cs', 'ca', ], ['-f', '-g', '-o'])
def _action_create_archive():
    ''' Analyze data files and archive results.
    '''
    from cerman.analyze_cm import create_archive as func
    logger.info('Loading and parsing saved data to archive')
    func(fpath, glob=pargs.glob, options=pargs.options)


@addToActions(['plot_simulation', 'ps',], ['<KIND>', '-f', '-g', '-o'])
def _action_plot_simulation():
    ''' Create plots from one or more simulations.
    '''
    from cerman.analyze_cm import plot_simulation as func
    logger.info(f'Creating simulation plot(s) of type: {pargs.kind}')
    func(fpath, plot_kind=pargs.kind, glob=pargs.glob, options=pargs.options)


@addToActions(['plot_iteration', 'pi',], ['<KIND>', '-r', '-f', '-g', '-o'])
def _action_plot_iteration():
    ''' Create plots from one or more iterations of one or more simulations.
    '''
    from cerman.analyze_cm import plot_iteration as func
    logger.info(f'Creating iteration plot(s) of type: {pargs.kind}')
    func(fpath, plot_kind=pargs.kind, glob=pargs.glob, idxes=pargs.idxes,
         options=pargs.options)


@addToActions(['plot_json', 'pj', ], ['-f', '-g', '-o'])
def _action_plot_json():
    ''' Create plots from one or more json-files.
    '''
    from cerman.analyze_cm import plot_json as func
    logger.info(f'Creating plot(s) from json file(s)')
    func(fpath, glob=pargs.glob, options=pargs.options)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       PLOT, SPECIFIC                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@addToActions(['plot_parameters', 'pp'], ['-f', '-g', '-o'])
def _action_plot_parameters():
    ''' Visual representation of differences in parameter values.
    Parsing json, log, input, or pkl files.
    '''
    from cerman.analyze_cm import plot_parameters as func
    logger.info('Creating parameter plot(s).')
    func(fpath, glob=pargs.glob, options=pargs.options)


@addToActions(['plot_stop_reason', 'psr'], ['-f', '-g', '-o'])
def _action_plot_stop_reason():
    ''' Visual representation of reason for simulation termination.
    '''
    from cerman.analyze_cm import plot_reason_for_stop as func
    logger.info('Creating stop reason plot(s).')
    func(fpath, glob=pargs.glob, options=pargs.options)


@addToActions(['plot_results', 'pc' ,'pr'], ['<KIND>', '-f', '-g', '-o'])
def _action_plot_results():
    ''' Plot results from several simulations in the same plot.
    '''
    from cerman.analyze_cm import plot_results as func
    logger.info('Creating combination plots.')
    func(fpath, plot_kind=pargs.kind, glob=pargs.glob, options=pargs.options)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       OTHER                                                         #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@addToActions(['create_movie', ], ['-f PATH', '-n FPS', '-o OUTPATH'])
def _action_create_movie():
    ''' Create a movie by combining images.
    '''
    from cerman.analyze.movie import create_movie as func
    logger.info('Plotting movie')
    func(fpath, fps=pargs.number, fpath_out=pargs.options)


@addToActions(['json', ], ['<KIND>', '-f', '-g'])
def _action_json():
    ''' Create/load json file. `update` or `append` globbed json files.
    '''
    import cerman.tools
    logger.info('Creating or modifying json file: "{}"'.format(fpath))

    opath = fpath  # outpath
    fpaths = cerman.tools.glob_fpath(fpath, glob=pargs.glob)

    if Path(opath).is_dir():
        logger.error(f'Error! Not a file, {opath}')

    elif not fpaths:
        logger.error(f'Error! No file(s) found, {fpath} + {pargs.glob}')

    else:
        if Path(opath).exists():
            # init from given file
            odata = cerman.tools.json_load(opath)

        else:
            # init from globbed opath
            odata = cerman.tools.json_load(fpaths[0])
            fpaths = fpaths[1:]  # makes looping later simpler

        if type(odata) is list:
            for fp in fpaths:
                logger.debug(f'Adding {str(fp)}')
                odata += cerman.tools.json_load(fp)  # append extra list
        else:
            # assume dictionaries
            for fp in fpaths:
                logger.debug(f'Updating with {str(fp)}')
                odata.update(cerman.tools.json_load(fp))

        logger.info(f'Dumping data to {str(opath)}')
        cerman.tools.json_dump(data=odata, fpath=opath)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       HELP                                                          #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@addToActions(['parameter_info', ], ['-o'])
def _action_parameter_info():
    ''' Print information on simulation parameters.
    All are listed unless keys or abbreviations are given.
    '''
    from cerman.analyze.sim_parameters import get_key_abbr_info
    logger.info('Information on simulation parameters:')
    logger.info(get_key_abbr_info(keys=pargs.options))

@addToActions(['results_info'], ['-o'])
def _action_results_info():
    ''' Print information on keys for the results of simulations.
    All are listed unless keys or abbreviations are given.
    '''
    from cerman.analyze.combination.results_repr import get_result_key_info
    logger.info('Information on result parameters:')
    logger.info(get_result_key_info(keys=pargs.options))

@addToActions(['rcparams_info'])
def _action_rcparams_info():
    ''' Print information how to change matplotlib's rcParams.
    '''
    from cerman.analyze.cerman_rc_params import CermanRCParams
    logger.info("Options to change matplotlib's rcParams:")
    CermanRCParams.opt2help()

@addToActions(['help'])
def _action_help():
    ''' Print all help info!
    '''

    def _pause():
        # wait for user
        out = input('Press Enter to continue...')
        if out == 'q':
            raise SystemExit

    # print parser options
    parser.print_help()
    _pause()

    # build output, line by line
    lines = []
    lines += ['']
    lines += ['']
    lines += ['Available actions']
    lines += ['=================']
    lines += ['']

    # the longest action name needed to define the indent
    keys = [k for k, a in Action.all_actions.items()]
    longest = max(len(k) for k in keys)

    # terminal width needed to chunk help-text, height for chunking pause
    tcol, trow = shutil.get_terminal_size()
    mw = tcol - longest - 3  # max width text

    def _chunkyfy(text, mw=mw, indent=longest + 3):
        # add left indent and add newline at appropriate whitespace
        out = ''
        while (len(text) > mw):
            _tmp = text[:mw].rpartition(' ')[0]  # get chunk
            text = text[len(_tmp) + 1:]          # remove chunk
            out += ' ' * indent + _tmp + '\n'    # add chunk
        out += ' ' * indent + text               # add remaining chunk
        return out

    # add help text from each action
    for key, action in Action.all_actions.items():
        msg = f'''{key.upper(): >{longest}} : '''    # add name of action
        msg += f'''{action.keys[-1]} '''             # add action shorthand
        if action.opts:
            msg += '[' + ', '.join(action.opts) + ']'  # add options
        lines += [msg]
        lines += [_l
                  for line in action.help.splitlines()
                  for _l in _chunkyfy(line).splitlines()
                  ]
        lines += ['']

    # print help with added pauses at appropriate places
    for i, line in enumerate(lines):
        logger.info(line)
        if (i + 1) % (trow - 2) == 0:
            _pause()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       ENTRY POINT                                                   #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def entry_point():
    ''' Function to invoke when running this file as main.'''

    # hack: pargs and fpath are automagically passed to `actions`
    global pargs
    global fpath

    # time program, shown in debug-log
    time_start = datetime.datetime.now()

    # set these globals, using the command line arguments
    pargs = parse_arguments()
    fpath = pargs.fpath

    # set up logging
    configure_logger()
    logger.debug(f'Start time {str(time_start)}')

    # perform the appropriate action
    Action.call(key=pargs.action)

    # show elapsed time
    time_end = datetime.datetime.now()
    logger.debug(f'End time {str(time_end)}')
    logger.debug(f'Elapsed time {str(time_end - time_start)}')


if __name__ == '__main__':
    entry_point()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       END OF FILE                                                   #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
