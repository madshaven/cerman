#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Tools shared by the software.

    Dump/load to/from file.
    Search/glob for files.
    Wrap functions to run on several files.
    Other shared tools.
'''

from pathlib import Path
import logging
import subprocess
import simplejson as json
import pickle
from collections import OrderedDict
import multiprocessing as mp
import csv

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Make json floats easier readable
json.encoder.FLOAT_REPR = lambda o: format(o, '.4g')


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       READ/WRITE                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def json_load(fpath):
    '''Load a json formatted file from `fpath`.

    Parameters
    ----------
    fpath :     str or Path
                path to file

    Returns
    -------
    data :      ordered dict
                loaded dictionary
    '''

    logger.log(5, f'Loading: {fpath}')

    fpath = Path(fpath)
    with fpath.open('r') as f:
        data = json.load(f, object_pairs_hook=OrderedDict)

    return data


def json_loads(s):
    '''Load a json string, return a dictionary.

    Parameters
    ----------
    json_string :     str
                      json string

    Returns
    -------
    data :      ordered dict
                loaded dictionary
    '''

    logger.log(5, f'Reading json-string')
    data = json.loads(s, object_pairs_hook=OrderedDict)

    return data


def json_dump(data, fpath, mode='w', sort_keys=False):
    '''Dump data dictionary to a json formatted file at `fpath`.

    Parameters
    ----------
    data :      dict
                dictionary to dump
    fpath :     str or Path
                path to file
    mode :      str, optional
                file write mode
    '''

    logger.log(5, f'Dumping: {fpath}')

    fpath = Path(fpath)
    if not fpath.parent.is_dir():
        fpath.parent.mkdir()
    with fpath.open(mode=mode) as f:
        f.write(json_dumps(data))


def json_dumps(data, sort_keys=False):
    '''Return data dict as json-string.

    Parameters
    ----------
    data :      dict
                dictionary to dump

    Returns
    -------
    data :      str
                json formatted dictionary
    '''

    return json.dumps(
        data, sort_keys=sort_keys, indent=4, separators=(',', ': '))


def pickle_load_gen(fpath):
    '''Load a pickle file from `fpath`.

    Parameters
    ----------
    fpath :     str or Path
                path to file

    Yields
    ------
    data :      arbitrary
                data loaded by pickle
    '''

    logger.log(5, f'Loading: {fpath}')

    fpath = Path(fpath)
    with fpath.open('rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def pickle_load(fpath):
    '''Load a pickle file from `fpath`.

    Parameters
    ----------
    fpath :     str or Path
                path to file

    Returns
    -------
    data :      arbitrary
                data loaded by pickle
    '''

    return next(pickle_load_gen(fpath))


def pickle_dump(data, fpath, mode='wb'):
    '''Dump data to pickle file `fpath`.

    Parameters
    ----------
    data :      dict
                dictionary to dump
    fpath :     str or Path
                path to file
    mode :      str
                file write mode
    '''

    logger.log(5, f'Dumping: {fpath}')

    fpath = Path(fpath)
    if not fpath.parent.is_dir():
        fpath.parent.mkdir()
    with fpath.open(mode=mode) as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def csv_load(fpath):
    '''Load a csv file from `fpath`.

    Parameters
    ----------
    fpath :     str or Path
                path to file

    Returns
    -------
    data :      list
                data loaded by csv
    '''

    logger.log(5, f'Loading: {fpath}')

    fpath = Path(fpath)
    with fpath.open('r', newline='') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]

    return rows


def csv_dump(rows, fpath, mode='w'):
    '''Dump rows to csv file `fpath`.

    Parameters
    ----------
    rows :      list
                list of rows to dump
    fpath :     str or Path
                path to file
    mode :      str
                file write mode
    '''

    logger.log(5, f'Dumping: {fpath}')

    fpath = Path(fpath)
    if not fpath.parent.is_dir():
        fpath.parent.mkdir()
    with fpath.open(mode=mode, newline='') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def csv_dump_dict_list(data, fpath, mode='w'):
    '''Dump data to csv file `fpath`.

    Parameters
    ----------
    data :      list[dict]
                list of dict dump
    fpath :     str or Path
                path to file
    mode :      str
                file write mode
    '''

    logger.log(5, f'Dumping: {fpath}')

    rows = []
    for d in data:
        kv_list = [[k, v] for k, v in d.items()]
        row = [i for kv in kv_list for i in kv]
        rows.append(row)

    csv_dump(rows, fpath, mode=mode)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       SEARCH/GLOB                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_fpath(fpath, folder=None, head=None, tail=None, ext=None):
    '''Return a path, create folder if required.

    Parameters
    ----------
    fpath  : str or Path
             output is based on the stem of this file
    head   : str
             added to the beginning of the file name
    tail   : str
             added to the end of the file name
    folder : str
             folder to add plot to
    ext    : str
             plot extension

    Returns
    -------
    file path : str
                fpath.parent / folder / name
    '''

    # evaluate input
    fpath = Path(fpath)
    name = fpath.stem
    if head is not None:
        name = '{}_{}'.format(head, name)
    if tail is not None:
        name = '{}_{}'.format(name, tail)
    if ext is not None:
        name = '{}.{}'.format(name, ext)
    # else:  # include this to keep suffix
    #     name = '{}{}'.format(name, fpath.suffix)
    if folder is None:
        folder = '.'

    # update fpath
    fpath = fpath.parent / folder / name

    # creating directory
    if not fpath.parent.is_dir():
        logger.debug('Created folder {}'.format(str(fpath.parent)))
        fpath.parent.mkdir()

    logger.log(5, 'Returning path: {}'.format(str(fpath)))
    return str(fpath)


def glob_fpath(fpath=None, glob=None, pre=None, post=None, rglob=None):
    '''Glob for list of Paths

    Notes
    -----
    If `glob` is specified:
        - In current folder, if fpath is None
        - In fpath, is fpath is a folder
        - In fpath's folder, if fpath is a file
    If `glob` is None:
        - Return fpath, if fpath is a file
        - Glob for `*` is fpath is a folder

    Parameters
    ----------
    fpath : str or Path
            file or folder path
    glob  : str
            center of globbing pattern
    pre   : str
            prefix of globbing pattern
    post  : str
            postfix of globbing pattern

    Returns
    -------
    lst(Path)
        list of file paths, empty if nothing is found
    '''

    msg = f'fpath={fpath}, glob={glob}, pre={pre}, post={post}, rglob={rglob}'
    logger.log(5, msg)

    def _glob(fpath, glob, pre=pre, post=post, rglob=rglob):
        # set defaults
        if glob is None:
            glob = '*'                              # center pattern
        if pre is None:
            pre = ''                                # pattern prefix
        if post is None:
            post = ''                               # pattern postfix
        if rglob is None:
            rglob = False                           #
        if glob[0] != '*':
            glob = '*' + glob                       # allow alternative start
        if glob[-1] != '*':
            glob = glob + '*'                       # allow alternative end

        if glob.startswith('*' + pre):
            pre = ''
        if glob.endswith(post + '*'):
            post = ''

        if fpath.is_dir():                        # glob for files
            pattern = '{}{}{}'.format(pre, glob, post)
            msg = f'Globbing for "{pattern}" in "{fpath}"'
            logger.debug(msg)
            if rglob:
                fpaths = sorted(fpath.rglob(pattern))
            else:
                fpaths = sorted(fpath.glob(pattern))
            logger.info('Found {} file(s).'.format(len(fpaths)))

        else:
            fpaths = []                             # wrong input
            msg = 'Not a file or a directory: {}'.format(str(fpath))
            logger.warning(msg)
            msg = 'Current directory: {}'.format(str(Path('.').resolve()))
            logger.debug(msg)

        return fpaths

    if False:  # for symmetry reasons...
        pass

    # glob in current folder
    elif (fpath is None) and (glob is not None):
        fpath = Path('.')
        fpaths = _glob(fpath, glob)

    # glob in folder (or file.parent)
    elif (fpath is not None) and (glob is not None):
        fpath = Path(fpath)
        if fpath.is_file():
            fpath = fpath.parent
        fpaths = _glob(fpath, glob)

    # return path if file (else glob in folder?)
    elif (fpath is not None) and (glob is None):
        fpath = Path(fpath)
        if fpath.is_file():
            logger.info(f'Found 1 file: {fpath}')
            fpaths = [fpath]    # convert file to list

        elif fpath.is_dir():  # glob in folder
            fpaths =  _glob(fpath, glob)

        else:
            msg = f'Not a file or folder "{fpath}".'
            logger.warning(msg)
            fpaths = []

    # return nothing
    elif (fpath is None) and (glob is None):
        msg = 'Please specify a file or a glob-pattern.'
        logger.info(msg)
        fpaths = []

    else:
        logger.error('Implementation error!')
        fpaths = []

    # Show files
    msg = 'Files found:' + '\n'.join([str(fpath) for fpath in fpaths])
    logger.debug(msg)
    return fpaths


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       MAP FPATH FUCTIONS                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def map_fpaths(fpaths, func, info=None, worker_nos=None, **kwargs):
    '''Perform `func(fpath, **kwargs)` for fpath in fpaths.

    Can use multiprocessing to run in parallel.

    Parameters
    ----------
    fpaths  : [Path]
              paths
    func    : function
              function to wrap
    info    : str
              info.format(fpath=fpath, **kwargs) is printed
    worker_nos : int
              number of processes to use in parallel
              defaults to False, i.e. execution without multiprocessing
    '''

    len_fpaths = len(fpaths)
    if info is None:
        info = 'Processing {fpath}'  # set default message
    if len_fpaths > 1:               # add counter
        info = '{i} of {len_fpaths} - ' + info

    if worker_nos is None:
        worker_nos = False
    if worker_nos == -1:
        worker_nos = mp.cpu_count()  # get number of CPUs
        logger.log(5, f'Fount {worker_nos} CPUs.')

    kwargs_list = []
    for i, fpath in enumerate(sorted(fpaths)):
        _kwargs = dict(
            i=(i + 1),
            len_fpaths=len_fpaths,
            info=info,
            func=func,
            fpath=Path(fpath),
            **kwargs
            )
        kwargs_list.append(_kwargs)
    logger.log(5, f'kwargs_list: {kwargs_list}')

    # see function definitions below
    if worker_nos:
        # mapping function creating a pool of workers
        _map_fpaths_mp_pool(kwargs_list, worker_nos)
    else:
        # mapping function iterating over a loop
        _map_fpaths_loop(kwargs_list)


def _map_fpaths_mp_pool(kwargs_list, worker_nos):
    # used by map_fpaths for mapping function creating a pool of workers

    msg = 'Starting multiprocessing pool with {} workers.'
    logger.info(msg.format(worker_nos))

    try:
        with mp.Pool(worker_nos) as pool:
            # chunksize one makes sense for expensive functions
            # see definition of _map_fpaths_work below
            pool.map(_map_fpaths_work, kwargs_list, chunksize=1)

    except BaseException as ex:
        if type(ex) == KeyboardInterrupt:   # continuation possible
            logger.info('Keyboard interrupt.')

        else:                               # add info, if wanted
            logger.error('Execution of parallel pool failed.')
            ex_msg = 'An exception of type {0} occurred.\nArguments: {1!r}'
            logger.debug(ex_msg.format(type(ex).__name__, ex.args))
            logger.log(5, 'Stack trace:', exc_info=True)


def _map_fpaths_loop(kwargs_list):
    # used by map_fpaths for mapping function iterating over a loop

    logger.debug('Mapping function in a loop.')

    for kwargs in kwargs_list:   # loop over files

        try:
            # see definition of _map_fpaths_work below
            _map_fpaths_work(kwargs)             # perform function

        except BaseException as ex:
            if type(ex) == KeyboardInterrupt:   # continuation possible
                answer = input('Continue? (y/[n]): ') or 'no'
                if answer not in ['y', 'yes']:
                    break                       # ignore exception, continue

            else:                               # add info, if wanted
                logger.error('Execution of loop failed.')
                ex_msg = 'An exception of type {0} occurred.\nArguments: {1!r}'
                logger.debug(ex_msg.format(type(ex).__name__, ex.args))
                logger.log(5, 'Stack trace:', exc_info=True)


def _map_fpaths_work(kwargs):
    # used by map_fpaths
    # need to be pickle-able

    logger.log(5, f'_map_fpaths_work got: {kwargs}')
    # pop away to remove, what is left is for the function
    info = kwargs.pop('info')
    info_msg = info.format(**kwargs)  # do this before popping
    kwargs.pop('i')
    kwargs.pop('len_fpaths')
    func = kwargs.pop('func')
    fpath = kwargs.pop('fpath')

    logger.info(info_msg)
    try:
        func(fpath, **kwargs)

    except BaseException as ex:
        logger.error(info_msg + ' - FAILED')
        raise


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       GIT                                                           #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def git_revision():
    # Return the git revision as a string
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        GIT_REVISION = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL).strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'
    except subprocess.CalledProcessError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       VERSION                                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


try:
    from . import _version
    version = {}
    version['git_revision'] = _version.git_revision
    version['version'] = _version.version
    version['short_version'] = _version.short_version
    version['full_version'] = _version.full_version
except ImportError:
    version = {}
    version['git_revision'] = None
    version['version'] = None
    version['short_version'] = None
    version['full_version'] = None


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       DICT TOOLS                                                    #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def print_dict(dictionary):
    ''' Print the dictionary content.

    Parameters
    ----------
    data :      dict
                dictionary to dump
    '''

    # Find dictionary variable name
    # name = [i for i in globals() if globals()[i] is dictionary][0]
    # print('Printing dict - {}:'.format(name))

    longest = sorted(dictionary, key=len)[-1]
    length = len(longest) + 0

    for k, v in sorted(dictionary.items()):
        print('{0:{2:}} - {1}'.format(k, v, length))
    print('')


def print_dict_to_file(data, dname, fbase):
    ''' Save dictionary as `dname` to `fbase.py`.

    Parameters
    ----------
    data :      dict
                dictionary to dump
    dname :     str
                name of dictionary variable
    fbase :     str
                filename
    '''

    fpath = Path(fbase, '.py')

    longest = sorted(data, key=len)[-1]
    length = len(longest) + len(dname) + 4

    with fpath.open('w') as f:
        f.write('\n')
        f.write('# Automatically written file.')
        f.write('\n')
        f.write('\n')
        f.write('{} = {{}}'.format(dname))
        for key, val in sorted(data.items()):
            left = "{}['{}']".format(dname, key)
            if type(val) is str:
                right = " = '{}'".format(val)
            else:
                right = " = {}".format(val)
            f.write('\n{0:{2:}}{1}'.format(left, right, length))
        f.write('\n')
        f.write('\n')
        f.write('#')
        f.write('\n')


def compare_dicts(dict_lst):
    ''' Compare a list of dictionaries return the keys which differ.
    Perform a shallow test, comparing value by value.

    Parameters
    ----------
    dict_lst :  lst[dict]
                list of dictionaries to inspect

    Returns
    -------
    unequal_keys :  lst[str]
    '''

    unequal_keys = []
    for key in dict_lst[0]:
        for item in dict_lst:
            if item[key] != dict_lst[0][key]:
                unequal_keys.append(key)
                break

    return unequal_keys


def dict_of_list_to_list_of_dicts(dict_of_lists):
    '''Return a list of `n` dictionaries,
    given a dictionary of lists of length `n`.

    Parameters
    ----------
    dict_of_lists : dict[lst]
                    list of dictionaries to inspect

    Returns
    -------
    list_of_dicts : lst[dict]
    '''

    def get_dict(i):
        # correct type (i.e. OrderedDict or regular dict)
        return type(dict_of_lists)(
            (key, val_lst[i]) for key, val_lst in dict_of_lists.items())

    lengths = [len(val_lst) for key, val_lst in dict_of_lists.items()]
    length = max(lengths)
    incorrect_keys = [key
                      for key, val_lst in dict_of_lists.items()
                      if len(val_lst) != length
                      ]
    if incorrect_keys:
        msg = f'Error! Found keys of incorrect length: {incorrect_keys}'
        logger.error(msg)

    return [get_dict(i) for i in range(length)]


def update_dict(d, u):
    ''' Recursively update `d` with `u` for a deep update of a dictionary.

    Parameters
    ----------
    d : dict
        to be updated to
    u : dict
        to be updated from

    Returns
    -------
    d : dict
        updated dictionary
    '''

    for key, val in u.items():
        if type(val) is dict:
            r = update(d.get(key, {}), val)
            d[key] = r
        else:
            d[key] = u[key]
    return d



#
