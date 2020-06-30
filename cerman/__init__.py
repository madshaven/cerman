#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
A bit on how this software is organized.
The main function of the software is to simulate streamer propagation.

The script `run_cm` provides a nice command line
interface to most of the functionality of this software.

The module `simulation_input` is used to
create, modify, and validate simulation input parameters.
The module contains default parameters
along with tools to load, expand, and/or dump input files.

Simulations are performed by `simulate_cm`.
The classes contained in `simulate` are loaded to perform the calculation.
Some of the classes in `simulate` relies
on functions in `core` for hyperboles and geometry functions.

The output of simulations can be loaded and analyzed
by using the `analyze` module.
Most functions of this module can be accessed from `analyze_cm` as well.
'''


from sys import version_info

if not ((version_info.major == 3) and (version_info.minor >= 6)):
    msg = 'Python3.6 or above required. Python{}.{} is not supported.'
    msg = msg.format(version_info.major, version_info.minor)
    raise SystemError('Error! ' + msg)

# add submodules to the namespace
from . import analyze
from . import simulate
from . import core
from . import simulation_input


#
