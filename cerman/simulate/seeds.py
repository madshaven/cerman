#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Class to keep track of the seeds,
    i.e. all the anions, electrons and avalanches.

    This class is used to modify and get the properties of the seeds
    as well as classify, move and multiply them.
'''

# General imports
import numpy as np
import logging

# Import from project files
from .streamer_head import SHList

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# idea: refactor to a general class for particles
#       create a new class to manage the particles


class Seeds(object):
    def __init__(self, mu_e, mu_ion, efield_ion, efield_crit, time_step,
                 Q_crit, efield_dtype, micro_step_no
                 ):
        # initiate the seeds object without adding any seeds

        # define the logger
        self.logger = logging.getLogger(__name__ + '.Seeds')

        # set the dtype for electric field calculation
        if efield_dtype in ['sp', 'float32',]:
            self.efield_dtype = np.float32
        elif efield_dtype in ['dp', 'float64',]:
            self.efield_dtype = np.float64
        else:
            self.efield_dtype = np.float32
            msg = 'Can not use dtype "{}". Using "{}"" instead.'
            self.logger.warning(msg.format(efield_dtype, self.efield_dtype))

        # set the properties of the seeds
        self.mu_e         = mu_e        # electron mobility
        self.mu_ion       = mu_ion      # cation mobility
        self.efield_ion   = efield_ion  # electric field for ion detachment
        self.efield_crit  = efield_crit  # electric field for multiplication
        self.time_step    = time_step   # simulation max time step
        self.dt           = time_step   # simulation actual time step
        self.micro_step_no = micro_step_no   # micro time steps
        self.Q_crit       = Q_crit      # avalanche to streamer criterion

        # initialize the seed variables, start with no seeds
        self.pos = np.array([]).reshape(3, 0)   # [x, y, z] position
        self.Q = np.array([])                   # avalanche size
        self.clear()                            # initialize other variables

        # log initiation
        self.logger.debug('Initated Seeds')
        self.logger.log(5, 'Seeds.__dict__')
        for k, v in self.__dict__.items():
            self.logger.log(5, '  "{}": {}'.format(k, v))

    def clear(self):
        # Clear variables

        # rename for better readability below
        no = self.no
        nz = np.zeros  # return zeroes
        nf = lambda shape: np.zeros(shape, dtype=bool)  # return false shape

        # Should be explicitly reset
        self.no_added       = 0
        self.no_removed     = 0
        self.pos_to_append  = nz((3, 0))
        self.is_to_remove   = nz(no, dtype=bool)

        # Should be updated each iteration
        self.ds             = nz((3, no))   # last movement
        self.ds_abs         = nz(no)        # movement, absolute
        self.ds_max         = 0             # movement, maximum
        self.dQ             = nz(no)        # change in charge
        self.mu             = nz(no)        # mobility
        self.e_dir          = nz((3, no))   # electric field, unit vector
        self.e_vec          = nz((3, no))   # electric field, vector
        self.e_str          = nz(no)        # electric field, strength
        # bools
        self.is_in_streamer = nf(no)        # inside streamer
        self.is_critical    = nf(no)        # avalanche termination
        self.is_behind_roi  = nf(no)        # behind ROI
        self.is_avalanche   = nf(no)        # avalanche seed
        self.is_ion         = nf(no)        # electron seed
        self.is_electron    = nf(no)        # ion seed

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       DERIVED ATTIBUTES                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # The user need to ensure that relevant variables are updated
    # before getting these properties

    Q_max = property(lambda self: self.Q.max())
    dQ_max = property(lambda self: self.dQ.max())
    charge = property(lambda self: np.exp(self.Q).sum())
    charge_gen = property(lambda self: (self.dQ * np.exp(self.Q)).sum())
    charge_rem = property(lambda self: np.exp(self.Q_to_remove).sum())

    no = property(lambda self: self.pos.shape[1])  # total number
    no_to_append = property(lambda self: self.pos_to_append.shape[1])

    no_ion = property(lambda self: self.is_ion.sum())
    no_electron = property(lambda self: self.is_electron.sum())
    no_avalanche = property(lambda self: self.is_avalanche.sum())
    no_critical = property(lambda self: self.is_critical.sum())
    no_behind_roi = property(lambda self: self.is_behind_roi.sum())
    no_in_streamer = property(lambda self: self.is_in_streamer.sum())
    no_to_remove = property(lambda self: self.is_to_remove.sum())

    pos_ion = property(lambda self: self.pos[:, self.is_ion])
    pos_electron = property(lambda self: self.pos[:, self.is_electron])
    pos_avalanche = property(lambda self: self.pos[:, self.is_avalanche])
    pos_critical = property(lambda self: self.pos[:, self.is_critical])
    pos_behind_roi = property(lambda self: self.pos[:, self.is_behind_roi])
    pos_in_streamer = property(lambda self: self.pos[:, self.is_in_streamer])
    pos_to_remove = property(lambda self: self.pos[:, self.is_to_remove])

    Q_ion = property(lambda self: self.Q[self.is_ion])
    Q_electron = property(lambda self: self.Q[self.is_electron])
    Q_avalanche = property(lambda self: self.Q[self.is_avalanche])
    Q_critical = property(lambda self: self.Q[self.is_critical])
    Q_behind_roi = property(lambda self: self.Q[self.is_behind_roi])
    Q_in_streamer = property(lambda self: self.Q[self.is_in_streamer])
    Q_to_remove = property(lambda self: self.Q[self.is_to_remove])

    dQ_ion = property(lambda self: self.dQ[self.is_ion])
    dQ_electron = property(lambda self: self.dQ[self.is_electron])
    dQ_avalanche = property(lambda self: self.dQ[self.is_avalanche])
    dQ_critical = property(lambda self: self.dQ[self.is_critical])
    dQ_behind_roi = property(lambda self: self.dQ[self.is_behind_roi])
    dQ_in_streamer = property(lambda self: self.dQ[self.is_in_streamer])
    dQ_to_remove = property(lambda self: self.dQ[self.is_to_remove])


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       UPDATE                                                    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Updating of bools should be called explicitly
    # since they may be expensive to calculate
    # or it has to be done at specific times
    # Maintain a comment to when they are updated!

    # `is_ion` is set during movement
    def update_is_ion(self, e_str=None):
        e_str = self.e_str if (e_str is None) else e_str
        self.is_ion = (e_str < self.efield_ion)
        return self.is_ion

    # ´is_electron´ is set during movement
    def update_is_electron(self, e_str=None):
        e_str = self.e_str if (e_str is None) else e_str
        self.is_electron = (e_str >= self.efield_ion)
        return self.is_electron

    # ´is_avalanche´ is set during movement
    def update_is_avalanche(self, e_str=None):
        e_str = self.e_str if (e_str is None) else e_str
        self.is_avalanche = (e_str >= self.efield_crit)
        return self.is_avalanche

    # Invoked in main simulation loop
    def update_is_critical(self, Q=None):
        Q = self.Q if (Q is None) else Q
        self.is_critical = (Q >= self.Q_crit)
        return self.is_critical

    # Invoked in main simulation loop
    def update_is_behind_roi(self, roi):
        self.is_behind_roi = roi.is_pos_behind(self.pos)
        return self.is_behind_roi

    # ´is_in_streamer´ is set during movement
    def update_is_in_streamer(self, heads):
        self.is_in_streamer = SHList(heads).is_inside(self.pos)
        return self.is_in_streamer

    def update_mu(self):
        self.mu = np.ones(self.no) * self.mu_ion
        self.mu[self.is_electron] = self.mu_e

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       OTHER                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def eprop(self, heads, pos=None):
        ''' Return `edir`, `estr`, `evec`, and `is_inside`.
            Convenience function to ensure correct dtype for seeds.
        '''
        if pos is None:
            pos = self.pos
        return SHList(heads).eprop(pos, dtype=self.efield_dtype)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       MOVE & MULTIPLY                                           #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _update_ds(self, e_str, e_vec):
        ''' Calculate movement. Update `ds`.
            Also update `is_electron`, `is_ion`, `mu`.
        '''
        self.is_electron = (e_str > self.efield_ion)
        self.is_ion = ~self.is_electron  # note: need to be somewhere
        self.mu = self.mu_ion * np.ones(self.no)
        self.mu[self.is_electron] = self.mu_e
        self.ds = e_vec * self.mu * self.dt

    def _update_ds_abs(self, thresh):
        ''' Calculate the max movement. Update `ds_abs` and `ds_max`.
            Give warning if any movement exceeds `thresh`.
        '''
        self.ds_abs = np.linalg.norm(self.ds, axis=0)
        self.ds_max = self.ds_abs.max()
        if self.ds_max > thresh:
            mask = (self.ds_abs > thresh)
            msg = 'Warning! {}x large movement!'.format(mask.sum())
            self.logger.warning(msg)
            self.logger.debug('Movements {}'.format(self.ds_abs[mask]))

    # note: previously used function
    def move_multiply_single_step(self, heads, calc_alpha):
        ''' Move all seeds and multiply all avalanches, single step. '''

        # treat seeds below the plane as if they were at the plane
        pos = self.pos.copy()
        mask = (self.pos[2, :] < 0)
        pos[2, mask] = 0

        # Find the field at electrons!
        self.e_dir, self.e_str, self.e_vec, self.is_in_streamer = (
            self.eprop(heads, pos))

        # Move seeds
        self._update_ds(self.e_str, self.e_vec)
        self._update_ds_abs(thresh=heads[0].rp)
        self.pos += self.ds

        # Update electron numbers
        self.dQ = calc_alpha(self.e_str) * self.ds_abs
        self.dQ[self.e_str < self.efield_crit] = 0
        self.dQ[self.pos[2, :] <= 0] = 0
        self.Q += self.dQ


    def move_multiply(self, heads, calc_alpha):
        ''' Move all seeds, multiply electron in avalanches.

        Parameters
        ----------
        heads :     SHList
                    list of streamer heads used for field calculation
        calc_alpha : CalcAlpha
                     class instance for calculating electron multiplication

        Notes
        -----
        Seeds are classified by the electric field strength.
        Avalanches are moved in an inner loop.
        Critical avalanches and collisions with the streamer
        are saved and these break the inner loop.
        Other seeds are moved according to the time spent in the inner loop.
        Long seed movements give warnings.
        Positions and charge numbers are updated.
        '''

        # treat seeds below the plane as if they were at the plane
        # this implies that seeds below plane will move straight upwards
        pos = self.pos.copy()
        mask = (self.pos[2, :] < 0)
        pos[2, mask] = 0

        # find the field
        self.e_dir, self.e_str, self.e_vec, self.is_in_streamer = (
            self.eprop(heads, pos))

        # classify the seeds
        self.update_is_electron()   # uses self.e_str
        self.update_is_ion()        # uses self.e_str
        self.update_is_avalanche()  # uses self.e_str

        # set all mobilities
        self.update_mu()            # uses self.is_electron

        # extract information about just the avalanches
        is_avalanche = self.is_avalanche
        pos_avalanche = self.pos[:, is_avalanche]
        Q_avalanche = self.Q[is_avalanche]
        e_str = self.e_str[is_avalanche]
        e_vec = self.e_vec[:, is_avalanche]
        e_dir = self.e_dir[:, is_avalanche]
        is_in_streamer = self.is_in_streamer

        time_spent = 0

        def _get_ds(e_vec, mu=None, time_step=None):
            # return the change in avalanche positions
            if mu is None:
                mu = self.mu
            if time_step is None:
                time_step = self.time_step
            ds = e_vec * mu * time_step
            ds_abs = np.linalg.norm(ds, axis=0)
            return ds, ds_abs

        def _warn_large_ds(ds_abs, threshold, what):
            ds_max = ds_abs.max()
            if (ds_max < threshold):
                return ds_max
            mask = (ds_abs > threshold)
            msg = f'Warning! {mask.sum()}x large {what} movement!'
            self.logger.warning(msg)
            self.logger.info(f'Max movement: {(ds_max * 1e6):.2g} um')
            self.logger.log(5, f'Movements {ds_abs[mask]}')
            return ds_max

        # skip this part when it is not needed
        if any(is_avalanche):

            # do one iteration with the field already calculated
            time_spent += self.time_step        # update the time
            ds, ds_abs = _get_ds(e_vec, mu=self.mu_e) # find the movements
            _warn_large_ds(ds_abs, threshold=heads[0].rp/10, what='avalanches')
            pos_avalanche += ds                 # update positions
            dQ = calc_alpha(e_str) * ds_abs     # get avalanche change
            dQ[pos_avalanche[2, :] < 0] = 0     # no multiplication for z<0
            Q_avalanche += dQ                   # update avalanche electrons

            # do the rest of the iterations
            for i in range(self.micro_step_no - 1):  # one is already done
                # end when critical avalanche is reached
                if any(Q_avalanche > self.Q_crit):
                    logger.debug('Critical avalanche!')
                    break
                # end on collision with streamer
                if any(is_in_streamer):
                    if i != 0:  # not needed (or allowed) the first loop
                        self.is_in_streamer[is_avalanche] = is_in_streamer
                    logger.debug('Collision with streamer!')
                    break

                # treat seeds below the plane as if they were at the plane
                pos_avalanche_tmp = pos_avalanche.copy()
                mask = (pos_avalanche[2, :] < 0)
                pos_avalanche_tmp[2, mask] = 0

                # find the field
                e_dir, e_str, e_vec, is_in_streamer = (
                    self.eprop(heads, pos_avalanche_tmp))

                time_spent += self.time_step        # update the time
                ds, ds_abs = _get_ds(e_vec, mu=self.mu_e) # find the movements
                _warn_large_ds(
                    ds_abs, threshold=heads[0].rp/10, what='avalanches')
                pos_avalanche += ds                 # update positions
                dQ = calc_alpha(e_str) * ds_abs     # get avalanche change
                dQ[pos_avalanche[2, :] < 0] = 0     # no multiplication for z<0
                Q_avalanche += dQ                   # update avalanche electrons

        else:  # no avalanches, should be a rare event
            time_spent = self.time_step * self.micro_step_no
            logger.debug('No avalanches!')

        # update avalanche numbers
        self.is_avalanche = is_avalanche  # should already be correct
        self.dQ[is_avalanche] = Q_avalanche - self.Q[is_avalanche]
        self.Q[is_avalanche] = Q_avalanche

        # get seed movements
        self.e_vec[:, is_avalanche] = 0  # do not move avalanches here
        ds, ds_abs = _get_ds(self.e_vec, time_step=time_spent) # find movements
        _warn_large_ds(ds_abs, threshold=heads[0].rp, what='seed')

        # update positions
        pos_old = self.pos.copy()   # save old position for ds calculation
        self.pos += ds              # add ds for all seeds
        self.pos[:, is_avalanche] = pos_avalanche  # set these explicitly

        # store actual movements
        self.ds = self.pos - pos_old
        self.ds_abs = np.linalg.norm(self.ds, axis=0)
        self.ds_max = self.ds_abs.max()

        # store the time spent here
        self.dt = time_spent

        # log results, as required
        logger.log(5, f'Number of avalanches, {is_avalanche.sum()}')
        logger.log(5, f'Time spent, {(time_spent * 1e9):.3f} ns')
        logger.log(5, f'Maximum movement, {(self.ds_max * 1e6):.3f} um')
        if any(is_avalanche):
            msg = f'Maximum avalanche growth, {(self.dQ.max()):.3f}'
        else:
            msg = f'Maximum avalanche growth, 0'
        logger.log(5, msg)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       APPEND/REMOVE                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def append(self, pos):
        ''' Append the given pos, and update vars as required.'''
        if not pos.size:
            self.logger.log(5, 'Tried to add zero seeds.')
            return
        assert pos.shape[0] == 3, 'pos.shape[0] == 3'
        Q = np.zeros(pos.shape[1])
        self.pos = np.hstack((self.pos, pos))
        self.Q = np.hstack((self.Q, Q))
        self.no_added += pos.shape[1]

    def remove(self, idx):
        ''' Remove pos and Q at the given indices.'''
        if idx.sum() == 0:
            self.logger.log(5, 'Tried to remove zero seeds.')
            return
        self.pos = self.pos[:, idx == 0]
        self.Q = self.Q[idx == 0]
        self.no_removed += idx.sum()

    def append_at_end(self, pos):
        ''' Appends the given pos to pos_to_append.'''
        assert pos.shape[0] == 3, 'pos.shape[0] == 3'
        pos = np.array(pos).reshape(3, -1)
        self.pos_to_append = np.hstack((self.pos_to_append, pos))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       CLEAN                                                     #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def clean(self):

        # Remove seeds and reset var
        self.remove(self.is_to_remove)

        # Append seeds and reset var
        self.append(self.pos_to_append)

        # Clear other variables
        self.clear()


#
