#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This file contains the default parameters for a simulation run.

    Expandable file
    ---------------
    The user may define an expandable input file.
    This is a JSON formatted file of only one dictionary.
    The keys should be the same as the ones below.
    From this file a number of new files is created.

    Keys containing lists will be expanded.
    If several keys contain lists then these are permutated.

    Similar runs add random seeds by incrementing the ones already defined.


    Simulation file
    ---------------
    This is a JSON formatted file of only one dictionary.
    The keys should be the same as the ones below.
    The values should be of the same type as the ones below.


    Simulation input
    ----------------
    Values defined below are overwritten
    by values defined in the simulation file.

    The input also contains these lists of keys:
    input_default;
    input_file;
    input_calculated;
    input_unused.

    The input is saved in each of the save files.

    The use of 'None' in this file implies that
    the value is to be derived.
    The use of 'None' in simulation input implies that
    the default value is to be used.
'''

# General imports
from collections import OrderedDict

''' consider:
    - INI format
    - a dict of all keys, with explanations (possibly a class):
        - enables an interface to get information about parameters
'''

# Dict for defaults
default_params = OrderedDict()
d = default_params  # short name
d['user_comment'] = 'Default parameters'   # Set by user

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       EXPERIMENTAL PROPERTIES                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
d['-- SECTION 1 --'] = 'EXPERIMENTAL PROPERTIES'

# The z-position of the needle
d['gap_size'] = 10e-3  # [m] Gap distance

# The tip point radius of the needle
d['needle_radius'] = 6e-6  # [m] Needle radius

# The voltage applied to the needle
d['needle_voltage'] = 100e3  # [V] Applied voltage
# idea: add possibility for voltage shape

# Townsend-Meek avalanche-to-streamer criterion
d['Q_crit'] = 23  # [number of electrons = exp(Q)]

# Method for calculating alpha (electron avalanche numbers)
d['alphakind'] = 'I2009'  # 'gas', 'I2009', 'A1991'

# Seeds number density (concentration)
# calculated from the liquids properties if set to 'None'
d['seeds_cn'] = 2e12  # [1/m3]

# Electric field strength for electron multiplication
# calculated from the liquids properties if set to 'None'
d['liquid_Ec_ava'] = 0.2e9  # [V/m]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       LIQUID                                                        #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# The name of the liquid
d['liquid_name'] = 'cyclohexane'

# The molar mass of the liquid
d['liquid_mass'] = 84.2  # [g/mol]

# The density of the liquid
d['liquid_density'] = 0.778  # [kg/l]

# The ionization potential of the liquid
d['liquid_IP'] = 9  # [eV]

# The lowest excited state of the liquid
d['liquid_e1'] = 6  # [eV]   First excited state

# The IP reduction constant of the liquid
d['liquid_beta_b'] = 50.8  # [eV]

# The relative permittivity of the liquid
d['liquid_R_perm'] = 2.02  # [1]

# The electron mobility of the liquid
d['liquid_mu_e'] = -4.5e-5  # [m^2/Vs]

# The ion mobility of the liquid
d['liquid_mu_ion'] = -3e-7  # [m^2/Vs]

# The ion conductivity of the liquid
d['liquid_sigma'] = 2e-13  # [1/ohm m]

# The threshold field for electron detachment in the liquid
d['liquid_Ed_ion'] = 1.0e6  # [V/m]

# The maximum ionization coefficient of the liquid
d['liquid_alphamax'] = 130e6  # [1/m]

# The inelastic scattering coefficient of the liquid
d['liquid_Ealpha'] = 1.9e9  # [V/m]

# The proportionality factor (for low-IP additives) of the liquid
d['liquid_k1'] = 2.8  # [eV]

# The critical net ionization coefficient of the liquid
# (minimum for electron multiplication)
d['liquid_alphacrit'] = 1e-2  # [1/m]

# Additive name and info
d['additive_name'] = 'DMA'

# Additive weight concentration
# 'cw', density and mass of liquid and additive is used to derive 'cn'
d['additive_cw'] = 0.0  # [wt %]

# Additive concentration in mole fraction
# derived from 'cw', overwrites 'cw' if specified
d['additive_cn'] = None  # [1]

# The molar mass of the additive
d['additive_mass'] = 121  # [g/mol]

# The density of the additive
d['additive_density'] = 0.956  # [kg/l]

# The ionization potential of the additive
d['additive_IP'] = 7.1  # [eV]

# The IP reduction constant of the additive
d['additive_beta_b'] = 40.7  # [eV]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       STREAMER PHYSICS                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
d['-- SECTION 2 --'] = 'STREAMER PHYSICS'

# Electrostatic streamer head tip radius
d['streamer_head_radius'] = 6e-6  # [m]

# Voltage gradient along streamer (minimum electric field in streamer channel)
d['streamer_U_grad'] = 0e6  # [V/m]

# Minimum distance for streamer heads (merge when closer)
d['streamer_d_merge'] = 25e-6  # [m]

# Threshold scaling for electrostatic shielding (remove heads with lower scale)
d['streamer_scale_tvl'] = 0.2  # [1]

# Enable (or disable) photoionization (4th mode)
d['streamer_photo_enabled'] = False  # [bool]

# Threshold electric field for photoionization
# may be derived from the liquids properties
d['streamer_photo_efield'] = None  # [V/m]

# Speed of 4th mode (photoionization)
d['streamer_photo_speed'] = 1e5  # [m/s]

# Mobility of streamer heads
# experimental feature, streamer heads moving in the electric field
d['streamer_repulsion_mobility'] = 0  # [m2/Vs]

# Z-offset of new streamer heads (compared to avalanche position)
# experimental feature, consider using "- streamer_head_radius"
d['streamer_new_head_offset'] = -0  # [m]

# RC-model relaxation time (tau0 = R(0) C(d))
d['rc_tau0'] = 1e-20  # [s]

# RC-model resistance model
# ['constant', 'linear']
d['rc_resistance'] = 'linear'

# RC-model capacitance model
# ['constant', 'hyperbole', 'plane', 'sphere', 'half_sphere']
d['rc_capacitance'] = 'constant'

# RC-model, threshold field for breakdown in streamer channel
d['rc_breakdown'] = 1e20  # [V/m]

# RC-model, factor for changing resistance upon breakdown
d['rc_breakdown_factor'] = 1e-20  # [1]

# Onsager model for conduction in streamer channel
# experimental feature, based on ion dissociation in liquids
d['rc_onsager'] = False  # [bool]

# Method for setting the potential of 'merged' streamer heads
# ['zero', 'existing', 'propagate', 'share_charge', 'final']
d['rc_potential_merged'] = 'propagate'

# Method for setting the potential of 'branched' streamer heads
# ['zero', 'existing', 'propagate', 'share_charge', 'final']
d['rc_potential_branched'] = 'share_charge'


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       SIMULATION PROPERTIES                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
d['-- SECTION 3 --'] = 'SIMULATION PROPERTIES'

# Simulation time step, for moving electrons and avalanches (inner loop)
d['time_step'] = 1e-12  # [s]

# Maximum time steps to perform in inner loop before moving anions
d['micro_step_no'] = 100  # [1]

# Seed used to initiate random number generator of numpy
# if `random_seed` is `None`, a random one is created
d['random_seed'] = None  # [#]

# Specify single or double precision for electric field calculations
d['efield_dtype'] = 'sp'  # ['sp', 'dp']


# Stop simulation, maximum iteration number
d['stop_iteration'] = 1e8     # [#]

# Stop simulation, minimum z-value
d['stop_z_min'] = 100e-6  # [m]

# Stop simulation, maximum number of streamer heads
d['stop_heads_max'] = 50     # [#]

# Stop simulation, minimum number of electrons
d['stop_seeds_no_electron'] = 0    # [#]

# Stop simulation, maximum simulation time
d['stop_sim_time'] = 10e-4  # [s]

# Stop simulation, maximum CPU time
d['stop_cpu_time'] = 1e6  # [s]

# Stop simulation, maximum CPU time, single iteration
d['stop_cpu_dt'] = 1e6  # [s]

# Stop simulation, maximum time between avalanches
d['stop_time_since_avalanche'] = 1e-7  # [s]

# Stop simulation, minimum streamer (average) speed
d['stop_speed_avg'] = 100  # [m/s]

# idea: time since growth (z_min change)
# idea: q_max_crit --> Low seed activity
# idea: add 'minimum'/'continue' criterion as well (min iterations, min time)

# ROI, initial radial size
d['roi_r_initial'] = 3e-3  # [m] initial rad

# ROI, roi radial growth by when a head is closer than this to the edge
d['roi_r_growth'] = 1e-4  # [m] radial growth

# ROI, the maximum radius
d['roi_r_max'] = 3e-3  # [m] maximum radius

# ROI, the extent of the ROI below (in front) of the leading head
d['roi_dz_below'] = 1e-3  # [m] extent below head

# ROI, the extent of the ROI above (behind) of the leading head
d['roi_dz_above'] = 1e-3  # [m] extent above head

# ROI, how to replace seeds
d['roi_replacement'] = 'distance' #
# distance -- move one ROI height, random xy position
# bottom -- move to bottom of ROI, random xy position
# random -- move to random position within ROI
# expanded -- move to expanded area (when growing, else bottom)
# density -- do not replace, but add new when the ROI moves instead
# idea: derive properties from gap size and electric field
# idea: create function to monitor density of seeds


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       INPUT & OUTPUT                                                #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
d['-- SECTION 4 --'] = 'INPUT & OUTPUT'
# Parameters used, or set, when loading a simulation file
# No lists here!

# Project working directory, filename, and path
# for dumping default parameters
# (usually derived from input file when expanding)
d['pwd'] = '.'
d['name'] = 'cmsim'
d['fpath'] = d['pwd'] + '/' + d['name'] + '.json000'

# Save simulation input to dedicated file
d['save_input'] = False

# Number of similar runs
# (random seed is extended for each run)
d['simulation_runs'] = 10

# Sequential start number for expanded files
# (useful for creating simulation series)
# note: a sequence is from 000 to max 999
d['seq_start_no'] = 0

# Define how parameters lists are permutated
d['permutation_order'] = 'random_seed, needle_voltage'

# Add all defaults to each input file when expanding file
d['add_defaults'] = False

# Expand file to a this folder (where to put the expanded files)
d['exp_folder'] = '.'

# Enable profiling of code (dumping of profile data)
d['profiler_enabled'] = False

# How often to dump save data to file
d['file_dump_interv'] = 500

# How often to display data on screen
d['display_interv'] = 500
# idea: add customizable display

# Logging
# note: 5 - deep debug, 10 - debug, 20 - info, 30 - warning

# Location of log file
d['log_folder'] = '.'

# Level of logging to file
d['log_level_file'] = 20

# Level of logging to console
d['log_level_console'] = 20


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       SAVE SPECIFICATION                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
d['-- SECTION 5 --'] = 'SAVE SPECIFICATION'

# Predefined save specifications are false by default
d['save_specs_enabled'] = {}
sse = d['save_specs_enabled']
sse['stat'] = True       # save most important info, every step
# sse['streamer'] = True   # save details of the streamer, every 0.1 % of gap

# 'sa' - step, save all, every n'th step
# sse['sa0'] = True      # step, save all, every e0 step
# sse['sa1'] = True      # step, save all, every e1 step
# sse['sa2'] = True      # step, save all, every e2 step

# 'ta' - time, save all, every n'th second
# sse['ta10'] = True     # time, save all, every e-10 second
# sse['ta09'] = True     # time, save all, every e-9 second
# sse['ta08'] = True     # time, save all, every e-8 second
# sse['ta07'] = True     # time, save all, every e-7 second

# 'gp' - gap percent, save all, every n'th gap percent
# sse['gp1'] = True      # gap percent, save all, every percent
# sse['gp5'] = True
# sse['gp10'] = True
# sse['gp20'] = True

## other, see source code (simulate/save_spec)
# sse['shadow'] = True          # new streamer positions
# sse['initiation'] = True      # the initiation only
# sse['seeds_critical'] = True  # the first occurrences of critical seeds
# sse['rangeno'] = True         # a range of iteration numbers
# sse['gapfianl'] = True        # final part of propagation

# Custom save specs may be given as input
# see source code for more details
d['save_spec_dicts'] = {}
ssd = d['save_spec_dicts']


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       OTHER                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
d['-- SECTION 7 --'] = 'OTHER'
# set when simulation starts

# git hash where the package is placed
d['git_revision_local'] = None

# from _VERSION.py, set upon installation
d['git_revision_package'] = None

# from _VERSION.py, set upon installation
d['version'] = None

# from _VERSION.py, set upon installation
d['full_version'] = None

d['-- SECTION 8 --'] = 'END OF DEFAULTS'

#


def get_defaults():
    # return a copy of the default parameters
    return default_params.copy()


#
