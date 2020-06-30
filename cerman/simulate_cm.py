#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''This is the main streamer simulation script.
'''

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#                                                                     #
#       IMPORTS                                                       #
#                                                                     #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# General imports
import numpy as np
import sys          # Reading input arguments
import os           # Changing working directory
import logging      # Enable logging
import datetime     # Write time ended

# Import simulate (used for initiation)
from cerman.simulate.streamer_head import StreamerHead
from cerman.simulate.streamer import Streamer
from cerman.simulate.streamer_manager import StreamerManager
from cerman.simulate.rc import RC
from cerman.simulate.seeds import Seeds
from cerman.simulate.roi import ROI
from cerman.simulate.save_spec import SaveSpecs
from cerman.simulate.sim_data import SimData
from cerman.simulate.calc_alpha import CalcAlpha
from cerman.simulate.log_config import init_logger
from cerman.simulate.profiler import Profiler

# Import input tools
from cerman.simulation_input import input_tools

# idea: class for printing/logging custom info
# idea: class for stop conditions, increased flexibility
# idea: improve logging, log once function
# idea: improve logging, logging statistics


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#                                                                     #
#       INITIATE                                                      #
#                                                                     #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def initiate(sim_input):
    '''Initiate all classes needed for running a simulation.

    Parameters
    ----------
    sim_input : dict
        dictionary containing all input required for a simulation

    Returns
    -------
    sim_vars : dict
        dictionary containing all classes needed for simulation
    '''

    # Create logger
    logger = logging.getLogger('cerman')
    init_logger(
        logger            = logger,
        log_name          = sim_input['name'],
        log_folder        = sim_input['log_folder'],
        log_level_file    = sim_input['log_level_file'],
        log_level_console = sim_input['log_level_console'],
        data              = sim_input,
        )

    # Create profiler
    profiler = Profiler(
        folder  = sim_input['pwd'],
        name    = sim_input['name'],
        enabled = sim_input['profiler_enabled']
        )

    # Set parameters for calculation of alpha
    calc_alpha = CalcAlpha(
        # todo: fix so error occurs if not available input
        kind = sim_input.get('alphakind', None),
        ea   = sim_input.get('liquid_Ealpha', None),
        am   = sim_input.get('liquid_alphamax', None),
        ka   = sim_input.get('liquid_k1', None),
        il   = sim_input.get('liquid_IP', None),
        ia   = sim_input.get('additive_IP', None),
        cn   = sim_input.get('additive_cn', None),
        )

    # Needle
    needle = StreamerHead(
        pos = [0, 0, sim_input['gap_size']],
        rp  = sim_input['needle_radius'],
        U0  = sim_input['needle_voltage'],
        k   = 1,
        )

    # Streamer
    streamer = Streamer()
    streamer.append(needle)  # the needle is treated like a streamer head
    streamer.clean()  # since the needle is not an avalanche

    # StreamerManager
    streamer_manager = StreamerManager(
        streamer = streamer,
        head_rp  = sim_input['streamer_head_radius'],
        U_grad   = sim_input['streamer_U_grad'],
        d_merge  = sim_input['streamer_d_merge'],
        origin   = needle,
        scale_tvl    = sim_input['streamer_scale_tvl'],
        photo_enabled = sim_input['streamer_photo_enabled'],
        photo_efield = sim_input['streamer_photo_efield'],
        photo_speed  = sim_input['streamer_photo_speed'],
        repulsion_mobility = sim_input['streamer_repulsion_mobility'],
        new_head_offset = sim_input['streamer_new_head_offset'],
        efield_dtype = sim_input['efield_dtype'],
        )

    # RC model
    rc = RC(
        origin       = needle,
        U_grad       = sim_input['streamer_U_grad'],
        tau0         = sim_input['rc_tau0'],
        resistance   = sim_input['rc_resistance'],
        capacitance  = sim_input['rc_capacitance'],
        breakdown    = sim_input['rc_breakdown'],
        breakdown_factor = sim_input['rc_breakdown_factor'],
        onsager      = sim_input['rc_onsager'],
        potential_merged = sim_input['rc_potential_merged'],
        potential_branched = sim_input['rc_potential_branched'],
        )
    streamer_manager.rc = rc

    # ROI
    roi = ROI(
        r  = sim_input['roi_r_initial'],
        rc = sim_input['roi_r_growth'],
        rm = sim_input['roi_r_max'],
        z  = sim_input['gap_size'],
        dz0 = sim_input['roi_dz_below'],
        dz1 = sim_input['roi_dz_above'],
        cn  = sim_input['seeds_cn'],
        replacement = sim_input['roi_replacement'],
        )

    # Seeds
    seeds = Seeds(
        mu_e           = sim_input['liquid_mu_e'],
        mu_ion         = sim_input['liquid_mu_ion'],
        efield_ion     = sim_input['liquid_Ed_ion'],
        efield_crit    = sim_input['liquid_Ec_ava'],
        time_step      = sim_input['time_step'],
        Q_crit         = sim_input['Q_crit'],
        efield_dtype   = sim_input['efield_dtype'],
        micro_step_no  = sim_input['micro_step_no'],
        )

    # Create and add initial seeds
    np.random.seed(sim_input['random_seed'])
    new_seeds_pos = roi.create_pos()
    seeds.append_at_end(new_seeds_pos)
    seeds.clean()  # adds seeds and resets variables
    seeds.update_is_in_streamer(streamer.heads)
    roi.manage_in_streamer(seeds)  # find seeds within streamer
    seeds.clean()  # replace seeds in streamer and reset variables

    # Save specifications
    save_specs = SaveSpecs(
        save_specs_enabled = sim_input['save_specs_enabled'],
        save_spec_dicts    = sim_input['save_spec_dicts'],
        folder             = sim_input['pwd'],
        name               = sim_input['name'],
        data               = sim_input,
        )

    # Simulation data
    sim_data = SimData()
    sim_data.set_stop_conditions(sim_input)
    sim_data.set_array_getters(streamer, seeds)
    sim_data.set_dict_getters(roi, streamer, seeds)

    # Sim variables
    sim_vars = {}
    sim_vars['logger']           = logger
    sim_vars['profiler']         = profiler
    sim_vars['calc_alpha']       = calc_alpha
    sim_vars['needle']           = needle
    sim_vars['streamer']         = streamer
    sim_vars['streamer_manager'] = streamer_manager
    sim_vars['roi']              = roi
    sim_vars['seeds']            = seeds
    sim_vars['save_specs']       = save_specs
    sim_vars['sim_data']         = sim_data
    sim_vars['sim_input']        = sim_input

    # Add initial data to sim_data
    sim_data.update_data(sim_vars)
    sim_data.extract_data(sim_vars)
    sim_data.data['no'] = 0
    sim_data.data['sim_time'] = 0
    sim_data.data['continue'] = True

    # Save initial data
    save_specs.update_save_now(sim_data.data)
    save_specs.append_data(sim_data.data)
    save_specs.dump_data()

    logger.info('Simulation variables initiation - COMPLETE.')
    return sim_vars


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#                                                                     #
#       SIMULATION LOOP                                               #
#                                                                     #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def simulation_loop(sim_vars):
    '''Perform one iteration of the simulation.

    The main steps are:
    - Update seeds
    - Photoionization
    - Manage new heads
    - Manage streamer (existing heads)
    - Update ROI (seeds)
    - Evaluate iteration

    Parameters
    ----------
    sim_vars : dict
        dictionary containing all classes needed for simulation
    '''

    logger           = sim_vars['logger']
    profiler         = sim_vars['profiler']
    calc_alpha       = sim_vars['calc_alpha']
    needle           = sim_vars['needle']
    streamer         = sim_vars['streamer']
    streamer_manager = sim_vars['streamer_manager']
    roi              = sim_vars['roi']
    seeds            = sim_vars['seeds']
    save_specs       = sim_vars['save_specs']
    sim_data         = sim_vars['sim_data']
    sim_input        = sim_vars['sim_input']

    # add one, since iteration no is updated at end of loop
    logger.iteration_no = sim_data.data['no'] + 1
    logger.log(5, 'Starting new iteration.')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                 #
    #       UPDATE SEEDS                                              #
    #                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Classify, move, and multiply seeds

    seeds.move_multiply(heads=streamer.heads, calc_alpha=calc_alpha)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                 #
    #       Photoionization                                           #
    #                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Move heads due to photoionization in front

    # note: creates new heads and removes old
    streamer_manager.move_photo(dt=seeds.dt)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                 #
    #       Electrostatic repulsion                                   #
    #                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Experimental feature!
    # note: creates new heads and removes old
    streamer_manager.move_repulsion(dt=seeds.dt)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                 #
    #       MANAGE NEW HEADS                                          #
    #                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Update critical seeds and create new heads
    seeds.update_is_critical()
    new_heads = streamer_manager.create_heads(seeds.pos_critical)
    new_heads = streamer_manager.trim_new_heads(new_heads)

    # Get merging and branching (mutates new_heads)
    merge_shl = streamer_manager.get_merging_heads(new_heads)
    branch_shl = streamer_manager.get_branching_heads(new_heads)

    # Set potential new heads
    # note: may modify potential of existing heads
    streamer_manager.rc.set_potential_merged(streamer, merge_shl)
    streamer_manager.rc.set_potential_branched(streamer, branch_shl)

    # Add the new heads to the streamer
    streamer.append(merge_shl)
    streamer.append(branch_shl)

    # Ensure needle is kept, always
    # idea: expand this section to allow voltage ramp for needle
    if needle not in streamer.heads:
        streamer.append(needle)
    needle.k = 1  # ensure correct setting for needle
    needle.U0 = sim_input['needle_voltage']

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                 #
    #       MANAGE STREAMER                                           #
    #                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Relax potentials, according to RC-model
    streamer_manager.rc.relax(streamer, needle, seeds.dt)

    # Remove streamer heads / trim streamer structure
    streamer_manager.remove_out_of_roi(roi)
    streamer_manager.remove_dist()
    streamer_manager.remove_nu_mat()
    streamer_manager.remove_nnls()

    # Set streamer head scale
    streamer_manager.set_scale_nnls()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                 #
    #       UPDATE ROI                                                #
    #                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # remove or redistribute critical seeds
    seeds.update_is_critical()
    roi.manage_critical(seeds)

    # remove or redistribute seeds within the streamer
    # seeds.update_is_in_streamer(streamer.heads)  # performed by `move_euler`
    roi.manage_in_streamer(seeds)

    # remove or redistribute seeds behind current ROI
    seeds.update_is_behind_roi(roi)
    roi.manage_behind_roi(seeds)

    # remove or redistribute seeds if ROI has moved
    roi.update_z(streamer.z_min)
    seeds.update_is_behind_roi(roi)
    roi.manage_changed_z(seeds)

    # create new seeds if ROI have increased in radius
    roi.update_r(streamer.r_max)
    roi.manage_changed_r(seeds)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                 #
    #       EVALUATE ITERATION                                        #
    #                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Update data
    sim_data.update_data(sim_vars)

    # Check whether the loop should continue
    sim_data.update_continue()

    # Update save now and save set for this iteration
    save_specs.update_save_now(sim_data.data)
    save_specs.update_save_set()

    # Update sim_data with data relevant for this iteration
    sim_data.extract_data(sim_vars, save_specs.save_set)

    # Extract relevant data from sim_data
    save_specs.append_data(sim_data.data)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #                                                                 #
    #       CLEAN UP                                                  #
    #                                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # idea: move these out and place at the end?
    # Save to file
    if (sim_data.data['no'] % sim_input['file_dump_interv'] == 0):
        save_specs.dump_data()
        profiler.dump_stats()

    # Add this to an external function.
    # Chose to view or not by sim_input
    if sim_data.data['no'] % sim_input['display_interv'] == 0:
        logger.info(sim_data.data['info'])
    # log every iteration when debugging
    elif logger.isEnabledFor(5):
        logger.log(5, sim_data.data['info'])

    streamer.clean()
    seeds.clean()
    roi.clean()
    streamer_manager.clean()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#                                                                     #
#       FINAL CLEAN UP                                                #
#                                                                     #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def finalize(sim_vars):
    '''Finalize a simulation, dump unsaved data, etc.

    Parameters
    ----------
    sim_vars : dict
        dictionary containing all classes needed for simulation
    '''

    logger           = sim_vars['logger']
    profiler         = sim_vars['profiler']
    save_specs       = sim_vars['save_specs']
    sim_data         = sim_vars['sim_data']
    sim_input        = sim_vars['sim_input']

    # Save any unsaved data to file
    save_specs.dump_data()

    # Dump stats
    profiler.dump_stats()

    logger.info(sim_data.data['info'])
    logger.info('')
    logger.info('SIMULATION ENDED')
    logger.info(sim_data.data['status'])
    logger.info('{}'.format(str(datetime.datetime.now())))
    # End of file
    for h in logger.handlers:
        h.close()
        logger.removeHandler(h)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#                                                                     #
#       SIMULATE                                                      #
#                                                                     #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def simulate(fpath):

    # Load simulation parameters
    sim_input = input_tools.read_input_file(fpath)

    # Change to correct working directory
    os.chdir(sim_input['pwd'])

    # Initiate variables
    sim_vars = initiate(sim_input)
    logger = sim_vars['logger']

    # Simulate in loop
    logger.info('STARTING SIMULATION.')
    logger.info('{}'.format(str(datetime.datetime.now())))
    while(sim_vars['sim_data'].data['continue']):

        try:
            # idea: consider splitting the loop into parts
            # the exception can be properly thrown out
            # while finalization may be handled
            # etc...
            simulation_loop(sim_vars)

        except BaseException as ex:
            msg = 'Error! An exception occurred in iteration {}.'
            msg.format(sim_data.data['no'])
            logger.error(msg)
            sim_vars['sim_data'].data['continue'] = False
            msg = 'An exception of type {0} occurred.\nArguments:{1!r}'
            msg = msg.format(type(ex).__name__, ex.args)
            sim_vars['sim_data'].data['info'] += '\n' + msg

            logger.debug(msg)
            logger.log(5, 'Stack trace:', exc_info=True)
            raise   # hack: improve later

    # Finalize simulation
    try:
        finalize(sim_vars)

    except BaseException as ex:
        # a message should be given if the finalization fails
        logger.info('Could not finalize the simulation.')

        msg = 'An exception of type {0} occurred.\nArguments:{1!r}'
        msg = msg.format(type(ex).__name__, ex.args)
        logger.debug(msg)
        logger.log(5, 'Stack trace:', exc_info=True)
        raise   # fixme: improve later?


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#                                                                     #
#       ENTRY POINT                                                   #
#                                                                     #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def entry_point():
    ''' Function to invoke when running this file as main.'''

    logger = logging.getLogger('cerman')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(20)  # info
    # logger.setLevel(5)  # for deep debug

    if len(sys.argv) == 1:
        msg = 'Please specify a file to simulate.'
        raise SystemExit(msg)

    elif len(sys.argv) == 2:    # simulate this file
        fpath = sys.argv[1]

    else:
        msg = ('Wrong number of input arguments. \n'
               'sys.argv: \n'
               '{}'.format(sys.argv))
        raise SystemExit(msg)

    try:
        simulate(fpath)

    except BaseException as ex:
        if type(ex) == KeyboardInterrupt:   # continuation possible
            logger.info('Keyboard interrupt. Simulation aborted.')

        else:                               # add info, if wanted
            logger.error('Error. Simulation terminated.')
            ex_msg = 'An exception of type {0} occurred.\nArguments: {1!r}'
            logger.debug(ex_msg.format(type(ex).__name__, ex.args))
            logger.log(5, 'Stack trace:', exc_info=True)


if __name__ == '__main__':
    entry_point()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       END OF FILE                                                   #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
