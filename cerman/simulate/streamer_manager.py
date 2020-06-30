#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This module contains a class for controlling list of StreamerHead objects.
    Its purpose is to create heads, find heads to add or remove, find scale.
'''

# General imports
import numpy as np
import logging
import scipy.special  # bessel function

# Import from project files
from ..core import coordinate_functions
from .streamer_head import SHList
from .streamer_head import StreamerHead

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

eps = np.finfo(float).eps  # 2.22e-16 for double
ooi = np.array([0, 0, 1]).reshape(3, -1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       STREAMER MANAGER                                              #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class StreamerManager(object):

    def __init__(self, streamer,
                 head_rp=1, U_grad=0, d_merge=0, origin=None,
                 scale_tvl=0,
                 photo_enabled=False, photo_efield=None, photo_speed=0,
                 repulsion_mobility=0, new_head_offset=0,
                 efield_dtype=np.float64,
                 ):

        # Initially set variables
        self.streamer = streamer        # the streamer to be managed
        self.head_rp = head_rp          # tip pint radius for new heads
        self.U_grad  = U_grad           # field within channel
        self.d_merge = d_merge          # merging distance for heads
        self.origin  = origin           # origin of streamer, needle
        self.scale_tvl    = scale_tvl   # threshold for removal of heads
        self.photo_enabled = photo_enabled      # photoionization enabled
        self.photo_efield = photo_efield        # field threshold for PI
        self.photo_speed  = photo_speed         # speed added for PI
        self.repulsion_mobility = repulsion_mobility
        self.new_head_offset = new_head_offset  # offset position of new heads
        self.efield_dtype = efield_dtype    # how to calculate electric field
        if self.efield_dtype == 'sp':
            self.efield_dtype = np.float32
        if self.efield_dtype == 'dp':
            self.efield_dtype = np.float64

        # Maintained variables, for nu and u matrix
        self.M_t  = []  # heads for M
        self.M_nu = np.zeros((0, 0))  # nu-mat
        self.M_u  = np.zeros((0, 0))  # u-mat

        # Reset variables (also to set them)
        self.clean()

        logger.debug('Initiated StreamerManager')
        logger.log(5, 'StreamerManager.__dict__')
        for k, v in self.__dict__.items():
            logger.log(5, '  "{}": {}'.format(k, v))

    def clean(self):
        self.heads_c = []    # Created
        self.heads_r = []    # Remove
        self.heads_a = []    # Append
        self.heads_m = []    # Merge
        self.heads_b = []    # Branch

    def update_M(self):
        ''' Update `M` variables, if required.'''
        if not simple_lst_cmp(self.streamer.heads, self.M_t):
            logger.log(5, 'Changed head list. Recalculating M.')
            self.M_t  = SHList(self.streamer.heads)  # ensure copy!
            self.M_nu = self.streamer.heads.calc_M_nu()
            self.M_u  = self.streamer.heads.calc_M_u(M_nu=self.M_nu)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       New heads                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def create_heads(self, pos, offset=None):
        '''Create new heads at given positions.'''

        # offset heads from avalanche position avalanches
        if offset is None:
            offset = ooi * self.new_head_offset
        else:
            offset = ooi * offset
        pos = pos + offset

        # create a list of new heads
        self.heads_c = SHList.from_lists(
            pos=pos,            # head position
            rp=self.head_rp,    # correct initial rp
            k=1,                # set scale later
            U0=0,               # set potential later
            dtype=self.efield_dtype,    # dtype for evec calculation
            )
        return self.heads_c

    def trim_new_heads(self, new_heads):
        ''' Remove merging and collided heads. Mutate and return new_heads.
        '''
        # Heads located within the streamer are removed.
        # Heads tagged for removal by merging algorithm are removed.

        # Warn if needed
        if len(new_heads) > 1:
            msg = 'Warning! More than one critical avalanche ({}).'
            msg = msg + ' Consider smaller time steps.'
            logger.warning(msg.format(len(new_heads)))

        # Is the new head within an old head? (collision)
        for head in list(new_heads):  # create a copy
            if self.streamer.heads.is_inside(head.pos):
                logger.log(5, 'Removed new head (inside existing)')
                new_heads.remove(head)

        # Is a streamer head close to this head? (merging)
        for head in list(new_heads):  # create a copy
            tmp_shl = SHList(self.streamer.heads + [head])
            to_remove = tmp_shl.to_remove_dist_recursive(dist=self.d_merge)
            if (head in to_remove):
                logger.log(5, 'Removed new head (merge)')
                new_heads.remove(head)

        return new_heads

    def get_merging_heads(self, new_heads):
        ''' Return a list of merging heads. Mutate new_heads.
        '''
        # Merging heads are new heads causing an existing to be merge.
        # Merging also if the nearest neighbor of a new head is within it.
        # idea: add merging mode to user inputs

        # initiate output
        merge_shl = SHList()

        # Is a streamer head close to this head? (merging)
        for head in list(new_heads):  # create a copy
            tmp_shl = SHList(self.streamer.heads + [head])
            to_remove = tmp_shl.to_remove_dist_recursive(dist=self.d_merge)

            if to_remove:  # note: (head in to_remove != True), removed above
                msg = 'Remove previous head(s) (merge) ({})'
                logger.log(5, msg.format(len(to_remove)))
                merge_shl.append(head)
                new_heads.remove(head)

        # Is nearest within this head? (merge)
        for head in list(new_heads):  # create a copy
            idx, dist = head.find_nearest(self.streamer.heads.pos)
            idx = int(idx)  # an array of idxes is returned above
            if head.is_inside(self.streamer.heads[idx].pos):
                logger.log(5, 'Remove previous head (inside new)')
                merge_shl.append(head)
                new_heads.remove(head)

        # Is a streamer head within this head? (absorption/propagation)
        for head in list(new_heads):  # create a copy
            if any(head.is_inside(self.streamer.heads.pos)):
                logger.log(5, 'Remove previous head (inside new)')
                # note: needed to avoid problems with charge sharing
                #       when the closest is also within
                #       this could also be handled elsewhere
                merge_shl.append(head)
                new_heads.remove(head)

        # Is the new head causing a head to be removed nnls?
        for head in list(new_heads):  # create a copy

            # calculate equipotential scales
            # store potential, calc scale, restore potential
            # alternatively, assume merge and set potential
            # # self.rc.set_potential_merged(self.streamer, head)
            # # self.rc.set_potential_branched(self.streamer, head)
            # note: what potential to use? equi? prev? final? propagate?

            # note: a new leading head usually removes a previous head
            #       using this method with "previous"
            #       failed to reduce potential at leading tip

            # create a new streamer with the head added
            tmp_shl = SHList(self.streamer.heads + [head])

            # get scales if the new head is added at equipotential
            U0_streamer = self.streamer.heads.U0    # store
            tmp_shl.U0 = 1                          # change
            k = tmp_shl.calc_scale_nnls()
            self.streamer.heads.U0 = U0_streamer    # change back

            # check for heads to be removed by nnls scale
            scale_tvl = min(self.scale_tvl, max(k))
            heads_r = [tmp_shl[i]
                       for i, ki in enumerate(k)
                       if ki < scale_tvl
                       ]

            # remove/manage new heads
            if heads_r:
                merge_shl.append(head)  #
                new_heads.remove(head)
            for head_r in heads_r:
                if head_r is head:
                    logger.log(5, 'Remove new head (nnls scale)')
                elif head_r in self.streamer.heads:
                    logger.log(5, 'Remove previous head (nnls scale)')
                else:
                    logger.warning('Warning! If-else clause missing!')

        # return list of merging heads
        if len(merge_shl) > 0:
            logger.log(5, 'Merging heads ({})'.format(len(merge_shl)))

        return merge_shl

    def get_branching_heads(self, new_heads):
        ''' Return a list of branching heads. Mutates new_heads.
        '''
        # Method: Branch if none of the above (not removed, not a merge).

        branch_shl = SHList()
        if new_heads:
            logger.log(5, 'New branches ({})'.format(len(new_heads)))
            if min(SHList(new_heads).d) < min(self.streamer.heads.d):
                logger.log(5, 'New leading branch')
            branch_shl = SHList(new_heads)
            new_heads = SHList()

        return branch_shl

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Set heads                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def set_scale_nnls(self):
        ''' Scale heads by nnls (non-negative least squares).'''

        # get and set scales
        self.update_M()  # updates if needed
        k = self.streamer.heads.calc_scale_nnls(M_u=self.M_u)
        self.streamer.heads.k = k

        # verify scales
        if (k == 0).sum() > 0:
            msg = 'Problem with scaling potential. {} head(s) set to zero.'
            logger.info(msg.format((k == 0).sum()))

        return k

    def set_scale_int(self):
        ''' Scale heads by integration of electric field.'''

        # get and set scales
        k = self.streamer.heads.calc_scale_int(no=None, dist=None)
        self.streamer.heads.k = k

        return k

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Remove heads                                              #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def remove_dist(self):
        ''' Remove heads are closer than given distance.'''

        self.heads_m = self.streamer.heads.to_remove_dist_recursive(
            dist=self.d_merge)

        if self.heads_m:
            msg = 'Found {} heads to merge.'
            logger.debug(msg.format(len(self.heads_m)))
            self.streamer.remove(self.heads_m)

        return self.heads_m

    def remove_nu_mat(self):
        ''' Remove heads located within other heads.'''

        self.update_M()  # updates if needed
        self.heads_r = self.streamer.heads.to_remove_nu_mat(
            M_nu=self.M_nu)
        if self.heads_r:
            msg = 'Found {} heads to remove by nu-mat.'
            logger.debug(msg.format(len(self.heads_r)))
            self.streamer.remove(self.heads_r)

        return self.heads_r

    def remove_nnls(self):
        ''' Remove heads based on nnls scaling.'''

        self.update_M()  # updates if needed
        k = self.streamer.heads.calc_scale_nnls(M_u=self.M_u)

        # do not remove the strongest
        scale_tvl = min(self.scale_tvl, max(k))
        self.heads_r = [self.streamer.heads[i]
                       for i, ki in enumerate(k)
                       if ki < scale_tvl
                       ]

        if self.heads_r:
            msg = 'Found {} heads to remove by k.'
            logger.debug(msg.format(len(self.heads_r)))
            self.streamer.remove(self.heads_r)

        return self.heads_r

    def remove_out_of_roi(self, roi):
        ''' Remove heads that are out of ROI.'''

        self.heads_r = []
        z1 = roi.z1
        # Tips lagging behind
        for head in self.streamer.heads:
            if head.pos[2] > z1:
                logger.debug('Removed head behind ROI.')
                self.heads_r.append(head)

        self.streamer.remove(self.heads_r)

        return self.heads_r

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Move heads                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def move_photo(self, dt):
        ''' Move self.streamer heads due to photoionization.

        Parameters
        ----------
        dt :    float
                time step used to calculate movement

        Notes
        -----
        Calculate the electric field strength at the tip of each head.
        Each head with an electric field strength,
        sufficiently lowering the ionization potential,
        is moved with a fixed speed.

        This is achieved by creating a new head and removing the old head.
        The potential of the "propagated head" is set according to
        the procedure for "merged" heads.


        Returns
        -------
        heads       : fast streamer heads, old - removed
        heads_new   : fast streamer heads, new - added
        '''

        if (self.photo_enabled is not True) or (self.photo_efield is None):
            return [], []

        # find fast heads
        estr = self.streamer.heads.estr(self.streamer.pos)
        idx = np.where(estr > self.photo_efield)[0]    # where returns a tuple
        heads = [self.streamer.heads[i] for i in idx]  # 4th mode heads
        pos_lst = [head.pos for head in heads]         # their positions
        pos = safe_hstack(pos_lst)                     # as array

        # create new heads
        ds = - ooi * self.photo_speed * dt
        pos_new = pos + ds
        heads_new = self.create_heads(pos_new)
        self.rc.set_potential_merged(self.streamer, heads_new)  # fix potential

        # update self.streamer
        self.streamer.append(heads_new)   # appending before removing
        self.streamer.remove(heads)       # to avoid "leading head" issue
        msg = 'Moved {} head(s), {:#0.3g} um'
        msg = msg.format(len(heads), float(ds[2] * 1e6))
        logger.log(5, msg)

        return heads, heads_new

    def move_repulsion(self, dt):
        ''' Move streamer heads due to electrostatic repulsion.

        Calculate the electric field at the center (d + rp) of each head.
        Exclude the current head from the calculation.
        Move the head, dr = E mu dt, i.e. create a new head at that position.
        '''

        mu = self.repulsion_mobility
        if mu is None:
            return
        if mu == 0:
            return

        heads = [h for h in self.streamer.heads if h is not self.origin]
        for head in heads:
            shl = SHList([h for h in self.streamer.heads if h is not head])

            pos = head.pos.copy()
            pos[2] = head.d + head.rp
            evec = shl.evec(pos)

            dr = evec * dt * mu
            ds = np.linalg.norm(dr, axis=0)

            new_pos = head.pos + dr
            new_head = self.create_heads(new_pos)[0]

            # copy properties
            new_head.k = head.k
            new_head.U0 = head.U0

            # append/remove
            self.streamer.remove(head)      # appending before removing
            self.streamer.append(new_head)  # to avoid "leading head" issue

            msg = 'Moved a head, {:#0.3g} um'.format(float(ds * 1e6))
            logger.log(5, msg)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       OTHER                                                         #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def lst_cmp(a, b):
    ''' Compare two lists.

    Parameters
    ----------
    a, b :  lst
            lists to compare

    Returns
    -------
    bool :  True if the two lists contains the same items in the same order.
    '''

    if len(a) != len(b):
        return False

    # Reversing the testing as differences are likely to be at the end
    return all(ai == bi for (ai, bi) in zip(reversed(a), reversed(b)))


def simple_lst_cmp(a, b):
    ''' Compare two lists.

    The method assumes that:
        - the lists were equal
        - old elements may be removed
        - only new elements may be appended
        - sequence of the lists may not be changed

    Parameters
    ----------
    a, b :  lst
            lists to compare

    Returns
    -------
    bool :  True if the two lists contains the same items in the same order.

    '''

    return (len(a) == len(b)) and (a[-1] == b[-1])


def safe_hstack(pos_lst):
    return np.hstack([np.zeros((3, 0))] + pos_lst)


#
