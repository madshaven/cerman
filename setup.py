#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Setup for Cerman.
'''

import setuptools
import subprocess
from pathlib import Path


CLASSIFIERS = '''\
Development Status :: 4 - Beta
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Programming Language :: Python :: 3
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: POSIX :: Linux
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Physics
'''

NAME                = 'cerman'
MAINTAINER          = 'Inge Madshaven'
MAINTAINER_EMAIL    = 'inge.madshaven@gmail.com'
DESCRIPTION         = 'Simulating streamer propagation in dielectric liquids'
with Path('./README.md').open() as f:
    LONG_DESCRIPTION = f.read()
URL                 = None
DOWNLOAD_URL        = None
LICENSE             = 'MIT'
CLASSIFIERS         = list(i for i in CLASSIFIERS.split('\n') if i)
AUTHOR              = 'Inge Madshaven'
AUTHOR_EMAIL        = 'inge.madshaven@gmail.com'
PLATFORMS           = ['MacOS', 'Linux']
MAJOR               = 0
MINOR               = 2
MICRO               = 0
IS_RELEASED         = True
VERSION             = f'{MAJOR:d}.{MINOR:d}.{MICRO:d}'
VERSION_FILE        = f'{NAME:s}/_version.py'


def _get_git_revision():
    # Return the git revision as a string
    try:
        cmd = ['git', 'rev-parse', 'HEAD']
        GIT_REVISION = subprocess.check_output(cmd).strip().decode('ascii')
    except OSError:
        GIT_REVISION = 'Unknown'

    return GIT_REVISION


def _get_git_revision_from_file():
    # what we want is this, without the import
    # from cerman._version import git_revision as GIT_REVISION

    with Path(VERSION_FILE).open('r') as f:
        for line in f:
            version = line[len("git_revision = '"):-2]
            break

    return version


if Path('.git').exists():
    # get local git revision, if possible
    GIT_REVISION = _get_git_revision()
elif Path(VERSION_FILE).exists():
    # read git revision from _version file
    GIT_REVISION = _get_git_revision_from_file()
else:
    GIT_REVISION = 'Unknown'


FULL_VERSION = VERSION
if not IS_RELEASED:           # add git revision to full_version

    FULL_VERSION += '.dev-' + GIT_REVISION[:7]


def _write_version_py():
    ''' Write version information to `./cerman/_version.py`. '''

    cnt = f'''\
# THIS FILE IS WRITTEN BY CERMAN SETUP.PY UPON INSTALLATION

short_version = '{VERSION:s}'
version = '{VERSION:s}'
full_version = '{FULL_VERSION:s}'
git_revision = '{GIT_REVISION:s}'
release = {str(IS_RELEASED):s}
if not release:
    version = full_version

#
'''
    with Path(VERSION_FILE).open('w') as f:
        f.write(cnt)


def setup_package():
    ''' Install cerman. '''

    _write_version_py()

    setuptools.setup(
        name=NAME,
        python_requires='>=3.6',
        install_requires=[
            'numpy',
            'simplejson',
            'matplotlib',
            'scipy',
            'statsmodels',
            ],
        version=VERSION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        url=URL,
        download_url=DOWNLOAD_URL,
        classifiers=CLASSIFIERS,
        platforms=PLATFORMS,
        packages=setuptools.find_packages(),
        package_data = {'': ['*.mplstyle']},
        provides=['cerman'],
        license=LICENSE,
        entry_points={
          'console_scripts': [
              'cerman = cerman.run_cm:entry_point',
              # idea: add simulate_cm
              ],
          },
        )


if __name__ == '__main__':
    setup_package()

#
