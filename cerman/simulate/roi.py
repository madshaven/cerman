#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Functions related to Region Of Interest, which is used to manage the seeds.
'''

# General imports
import logging
import numpy as np

# Import from project files
from ..core.coordinate_functions import random_cyl_pos

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ROI(object):
    def __init__(self, r, z, dz0, dz1, rc, rm, cn, replacement):
        ''' Contains and controls the Region Of Interest (ROI) parameters
        used to manage the positions of seeds.

        Parameters
        ----------
        r  : float
             cylinder radius
        z  : float
             center z position
        dz0 : float
             extent below z position
        dz1 : float
             extent above z position
        rc : float
             if r - head.r < rc ==> r += rc / 2
        rm : float
             maximum radius
        cn : float
             number concentration of seeds
        '''

        self.logger = logging.getLogger(__name__ + '.ROI')

        # distance, expanded_z, density, bottom
        # set these before updating z
        self.replacement = replacement
        self.replacement_mode = replacement
        if replacement == 'distance':
            self.replacement_region = 'expanded_z'
        elif replacement == 'expanded':
            self.replacement_region = 'expanded_z'
        else:
            self.replacement_region = replacement

        self.r0 = 0         #: float: inner radius
        self.r1 = r         #: float: outer radius
        self.dz0 = dz0      #: float: distance below
        self.dz1 = dz1      #: float: distance above
        self.z_min = dz0 + 1e-10    #: float: lowest position
        self.z_min = -1e10          #: float: lowest position
        self.rm = rm        #: float: maximum radius
        self.z = 0          #: float: "center" position
        self.z0 = 0         #: float: lower position of ROI
        self.z1 = 0         #: float: upper position of ROI
        self.update_z(z)
        self.z_prev = self.z
        self.z0_prev = self.z0
        self.z1_prev = self.z1
        self.cn = cn
        self.rc = rc
        self.r1_prev = r  #: float: previous outer radius
        self.expanded_z = False  #: bool: ROI expanded this iteration
        self.expanded_r = False  #: bool: ROI expanded this iteration

        self.logger.debug('Initiated ROI')
        self.logger.log(5, 'ROI.__dict__')
        for k, v in self.__dict__.items():
            self.logger.log(5, '  "{}": {}'.format(k, v))

    def clean(self):
        ''' Reset variables as required at end of iteration.'''
        self.expanded_z = False
        self.expanded_r = False

    def update_r(self, r_head):
        ''' Increase ROI if needed.'''

        # head close to edge AND radius below max
        if (self.r1 - self.rc < r_head) and (self.r1 < self.rm):
            # expand
            self.r1_prev = self.r1
            self.r1 += self.rc / 2
            self.expanded_r = True

            # reduce, if too large
            if self.r1 > self.rm:
                self.logger.info('ROI maximum r reached.')
                self.r1 = self.rm

            # warn if one expansion was not enough
            if (self.r1 - self.rc < r_head):
                self.logger.warning(
                    'ROI still too small after expansion. '
                    'Consider a change of parameters.')

    def update_z(self, z):
        ''' Update z-variables, if required.'''
        # todo: consider making this safe for removal of leading head

        # update if z has change and is not the minimum
        if (z != self.z) and (self.z != self.z_min):

            # store the previous values
            self.z_prev = self.z
            self.z0_prev = self.z0
            self.z1_prev = self.z1

            # update mid, bottom and top
            self.z = max(z, self.z_min)
            self.z0 = self.z - self.dz0
            self.z1 = self.z + self.dz1
            # self.z0 = max(1e-10, self.z0)  # avoid seeds at z=0

            # flag that ROI have been changed
            self.expanded_z = True

            # change mode when bottom is reached
            # if self.z == self.z_min:
            #     self.replacement_mode = 'bottom'
            #     self.replacement_region = 'bottom'


    def volume(self):
        ''' Return the ROI volume.'''
        return np.pi * (self.z1 - self.z0) * (self.r1**2 - self.r0**2)

    def is_pos_behind(self, pos):
        ''' Return `True` if the given position is within ROI.'''
        return (pos[2, :] > self.z1)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       CREATE SEEDS                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def create_pos(self, region='all', no=None, cn=None):
        ''' Create random positions within the ROI.

        Parameters
        ----------
        region :    str
                    Specifies where the positions should be generated:
                        - all: the entire ROI
                        - top: only the top plane
                        - bottom: only the bottom plane
                        - expanded_z: region between z0 and z0_prev
                        - expanded_r: the newly expanded region
        no :        int
                    Number of positions.
                    Only used when `cn` is not specified.
        cn :        float
                    number concentration of positions.
                    ROI concentration is default.

        Returns
        -------
        array (3, -1)
            positions
        '''

        # set defaults
        if (no is None) and (cn is None):
            cn = self.cn
        r0 = self.r0
        r1 = self.r1
        z0 = self.z0
        z1 = self.z1

        # set region specific variables
        if region in ['all', 'random']:
            pass  # defaults
        elif region == 'top':
            z0 = self.z1
        elif region == 'bottom':
            z1 = self.z0
        elif region == 'expanded_z':
            z1 = self.z0_prev
        elif region == 'expanded_r':
            r0 = self.r1_prev
        else:
            logger.error(f'The region "{region}" is not available')
            return np.zeros((3, 0))

        # create and return positions
        msg = f'Creating seeds at {region}, using '
        msg += f'r0={r0}, r1={r1}, z0={z0}, z1={z1}, no={no}, cn={cn}'
        self.logger.log(5, msg)
        new_pos = random_cyl_pos(r0=r0, r1=r1, z0=z0, z1=z1, no=no, cn=cn)
        return new_pos

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       MANAGE SEEDS                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def manage_seeds(self, seeds, is_to_be_managed, region=None):
        ''' Flags which seeds that should be removed
        and creates new positions to be added.

        Parameters
        ----------
        seeds :     class Seeds
                    the instance of seeds to be managed
        is_to_be_managed :  array(bool)
                    the seeds to be changed by this operation
        region :    str(optional)
                    the region where to place the new seeds
        '''

        # only replace seeds that are not already taken care of
        is_to_be_managed = (is_to_be_managed & ~seeds.is_to_remove)
        no = is_to_be_managed.sum()

        if (no > 0):  # check if anything needs to be done

            if (self.replacement_mode == 'density'):
                # do not replace seeds if the mode is density
                new_pos = np.array([]).reshape(3, 0)
            else:
                region = region or self.replacement_region
                new_pos = self.create_pos(region=region, no=no)
            if self.replacement_mode == 'distance':
                prev_pos = seeds.pos[:, is_to_be_managed].copy()
                new_pos[2, :] = prev_pos[2, :] - self.dz0 - self.dz1

            # tell seeds what to add and remove during cleaning step
            seeds.append_at_end(new_pos)
            seeds.is_to_remove |= is_to_be_managed

    def manage_in_streamer(self, seeds):
        # Remove and possibly replace seeds in within streamer heads
        self.manage_seeds(seeds, seeds.is_in_streamer)

    def manage_critical(self, seeds):
        # Remove and possibly replace critical seeds
        self.manage_seeds(seeds, seeds.is_critical)

    def manage_behind_roi(self, seeds):
        # Remove and possibly replace seeds out of ROI
        self.manage_seeds(seeds, seeds.is_behind_roi)

    def manage_changed_z(self, seeds):
        # Remove and possibly replace seeds out of ROI

        if self.expanded_z:  # only needed if z has changed
            # remove and possibly replace seeds
            self.manage_seeds(seeds, seeds.is_behind_roi)

            # add new seeds based on density
            if (self.replacement_mode == 'density'):
                pos = self.create_pos(region='expanded_z', no=None)
                seeds.append_at_end(pos)

    def manage_changed_r(self, seeds):
        # Add new seeds for expanded ROI
        if self.expanded_r:
            pos = self.create_pos(region='expanded_r')
            seeds.append_at_end(pos)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Other                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def to_dict(self):
        ''' Create a dictionary from class attributes.'''
        roi_dict = {}
        roi_dict['r0']       = self.r0
        roi_dict['r1']       = self.r1
        roi_dict['dz0']      = self.dz0
        roi_dict['dz1']      = self.dz1
        roi_dict['rm']       = self.rm
        roi_dict['z']        = self.z
        roi_dict['cn']       = self.cn
        roi_dict['z0']       = self.z0
        roi_dict['z1']       = self.z1
        roi_dict['volume']   = self.volume()
        roi_dict['rc']       = self.rc
        roi_dict['replacement'] = self.replacement
        return roi_dict

#
