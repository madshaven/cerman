#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Module containing tools to analyze simulations.
'''

# add submodules (folders) to the namespace
from . import iteration           # analyse iteration(s)
from . import simulation          # analyse simulation(s)
from . import combination         # analyse several simulations

# add submodules (files) to the namespace
from . import load_data           # load the contents of a save file
from . import tools               # support tools
from . import inspect_data        # inspect contents of a save file
from . import movie               # combine pictures to a move
from . import plotterXZ           # base class for plotting
from . import plotterXYRZ         # class for plotting
from . import cerman_rc_params    # settings for plotting
from . import sim_parameters      # represent simulation parameters in plots


#
