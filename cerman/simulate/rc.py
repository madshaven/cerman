#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Resistive-capacitive model implementation.

    The class have methods for setting potential of streamer heads,
    and for relaxing the potential each iteration.

    The class have methods to get RC-factors for
    resistance in channel, capacitance towards the plane,
    breakdown in channel, conduction due to dissociation.
'''

# General imports
import numpy as np
import logging
from scipy.special import iv as bessel_iv  # bessel function

# Import from project files
from ..core import coordinate_functions
from .streamer_head import SHList
from .streamer_head import StreamerHead

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

eps = np.finfo(float).eps  # 2.22e-16 for double


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#       RC                                                            #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

class RC(object):

    def __init__(self,
                 origin,        # usually the needle
                 tau0,          # tau = RCtau0
                 U_grad,        # minimum E-field within channel
                 resistance,    # how to model channel resistance
                 capacitance,   # how to model capacitance
                 breakdown,     # threshold for breakdown in channel
                 breakdown_factor,  # tau *= bdf, when there is a breakdown
                 onsager,       # if true, enable Onsager model
                 potential_merged,      # potential model to use
                 potential_branched,    # potential model to use
                 ):

        self.origin = origin
        self.U_grad = U_grad
        self.tau0 = tau0
        self.resistance = resistance
        self.capacitance = capacitance
        self.breakdown_threshold = breakdown
        self.breakdown_factor = breakdown_factor
        self.onsager = onsager
        self.potential_merged = potential_merged
        self.potential_branched = potential_branched

        logger.debug('Initiated RC')
        logger.log(5, 'RC.__dict__')
        for k, v in self.__dict__.items():
            logger.log(5, '  "{}": {}'.format(k, v))

    @staticmethod
    def _cap_factor_constant(heads):
        # constant capacitance -- no dependence on anything
        return np.ones_like(heads.d)

    @staticmethod
    def _cap_factor_plane(heads):
        # model each streamer heads as a parallel plate capacitor
        # scale by gap length
        return (1 / heads.d)

    @staticmethod
    def _cap_factor_hyperbole(heads, origin):
        # model each streamer heads as a hyperboloid capacitor
        den = 4 * heads.a / heads.rp
        return 1 / np.log(den)

    @staticmethod
    def _cap_factor_sphere(heads, origin):
        # model capacitance as an expanding sphere, see Crowley 2008
        d = origin.d
        rp = origin.rp
        z = heads.d
        r = (d + 2 * rp - z) / 2  # sphere radius
        return r * (1 + 0.5 * np.log(1 + r / z))

    @staticmethod
    def _cap_factor_half_sphere(heads, origin):
        # model capacitance as an expanding half-sphere, see Crowley 2008
        d = origin.d
        rp = origin.rp
        z = heads.d
        r = (d + rp - z)          # half sphere radius
        # note: the half sphere is about twice the size of the sphere
        return r * (1 + 0.5 * np.log(1 + r / z))

    def _get_cap_factor(self, heads, origin, cdm):
        # return unscaled capacitance of each head, given origin and model

        # choose model for capacitance towards plane
        if (cdm == 'constant') or (cdm == '1') or (cdm == 1):
            return self._cap_factor_constant(heads)
        elif cdm == 'plane':
            return self._cap_factor_plane(heads)
        elif cdm == 'hyperbole':
            return self._cap_factor_hyperbole(heads, origin)
        elif cdm == 'sphere':
            return self._cap_factor_sphere(heads, origin)
        elif cdm == 'half_sphere':
            return self._cap_factor_half_sphere(heads, origin)
        else:
            msg = 'Error! Unknown capacitance model: {}'
            logger.error(msg.format(cdm))
            raise SystemExit

    def get_cap_factor(self, heads, origin=None, cdm=None):
        # return capacitance of the heads, scaled by the needle capacitance
        if origin is None:
            origin = self.origin  # the needle
        if cdm is None:  # capacitance dependence model
            cdm = self.capacitance

        c_origin = self._get_cap_factor(
            heads=origin, origin=origin, cdm=cdm)
        c_heads = self._get_cap_factor(
            heads=heads, origin=origin, cdm=cdm)
        return c_heads / c_origin

    def get_res_factor(self, heads, origin=None, ldm=None):
        # return length/resistance dependence, scaled by the gap distance
        if origin is None:
            origin = self.origin  # the needle
        if ldm is None:  # length dependence model
            ldm = self.resistance

        # choose model for resistance in channel
        if ldm == 'constant':
            # constant resistance -- no dependence on anything
            return np.ones_like(heads.d)
        elif ldm == 'linear':
            # scale resistance with length of channel
            length = origin.dist_to(heads.pos)
            return length / origin.z
        else:
            msg = 'Error! Unknown resistance model: {}'
            logger.error(msg.format(ldm))
            raise SystemExit

    def get_breakdown_factor(self, heads, origin=None, bdt=None, bdf=None):
        # return low resistance if/where breakdown in channel
        if origin is None:
            origin = self.origin
        if bdt is None:  # breakdown threshold value
            bdt = self.breakdown_threshold
        if bdf is None:  # breakdown factor
            bdf = self.breakdown_factor

        length = origin.dist_to(heads.pos)
        estr = (origin.U0 - heads.U0) / (length + eps)  # eps for safe needle

        bd = np.ones_like(length)   # default to factor 1
        bd[estr > bdt] = bdf        # set to bdf for breakdown
        return bd

    def get_onsager_faktor(self, heads):
        # enhanced conductance from ion dissociation, see Gäfvert 1992

        # note: this model was implemented to demonstrate non-linear effects
        #       it is for a liquid, not for a gas/plasma
        #       the temperature and the permittivity could be changed later

        if not self.onsager:
            return np.ones_like(heads.d)

        # field in channel
        length = self.origin.dist_to(heads.pos)
        # eps for safe needle
        estr = (self.origin.U0 - heads.U0) / (length + eps)

        # standard parameters
        T = 293             # K
        kb_J = 1.381e-23    # J/K
        e0 = 8.85e-12       # F/m, vacuum permittivity
        er = 2 * e0
        ec = 1.6e-19        # C, elementary charge

        # calculate dissociation
        estr = estr + eps
        _n = ec**3 * estr                       # nominator
        _d = 16 * np.pi * er * T**2 * kb_J**2   # denominator * 2
        _b = np.sqrt(_n / _d)                   # sq(b/2) ==> 2sq(b/2)=sq(2b)
        _f = bessel_iv(1, 4 * _b) / (2 * _b)    # 4sq(b/2)=sq(8b)
        h = 1 / _f  # increased conductance implies lower tau-factor here

        return h

    def relax(self, streamer, needle, dt):
        ''' Calculate the time constant and
            relax the potential of each streamer head.
        '''

        # get factors for time constant
        _ld = self.get_res_factor(streamer.heads)
        _cd = self.get_cap_factor(streamer.heads)
        _bd = self.get_breakdown_factor(streamer.heads)
        _od = self.get_onsager_faktor(streamer.heads)

        # combine all the factors
        tau = self.tau0
        tau *= _ld      # channel length dependence
        tau *= _cd      # capacitance dependence
        tau *= _bd      # breakdown in channel?
        tau *= _od      # Onsager dissociation

        tau = np.minimum(tau, 1 / eps)      # ensure tau < inf
        tau = np.maximum(tau, eps)          # ensure tau > 0

        # final potential
        Uf = self.get_final_potential(streamer.heads)
        # potentials differences
        diff_prev = Uf - streamer.heads.U0
        diff_new = diff_prev * np.exp(- dt / tau)
        diff_diff = diff_prev - diff_new

        if diff_diff.max() > 100:
            msg = 'Relaxed potential, max {:.1f} kV'
            logger.log(5, msg.format(diff_diff.max() * 1e-3))

        # set relaxed potentials
        streamer.heads.U0 = Uf - diff_new

    def _set_potential(self, streamer, heads, model):
        ''' Modify the potential of the heads,
            and possibly the streamer,
            depending on the chosen model.
        '''
        if isinstance(heads, (StreamerHead,)):
            heads = [heads]
        heads = SHList(heads)  # ensure streamer head list
        if model == 'zero':         # set all potentials to 0
            heads.U0 = 0
        elif model == 'previous':   # use potential at current position
            U0 = streamer.heads.epot(heads.pos)
            heads.U0 = U0
        elif model == 'propagate':  # propagate charge
            self.propagate_charge(streamer, heads)
        elif model == 'share_charge':  # share charge
            self.share_charge(streamer, heads)
        elif model == 'final':      # relax fully
            U0 = self.get_final_potential(heads)
            heads.U0 = U0
        else:
            msg = 'Error! Unknown potential model! ({})'
            logger.error(msg.format(model))
            raise SystemExit

    def set_potential_merged(self, streamer, heads):
        # apply the correct model to set potential for merged heads
        self._set_potential(streamer, heads, model=self.potential_merged)

    def set_potential_branched(self, streamer, heads):
        # apply the correct model to set potential for branched heads
        self._set_potential(streamer, heads, model=self.potential_branched)

    def propagate_charge(self, streamer, heads):
        ''' Set potential of the heads by propagate charge
            from the nearest existing head.
        '''
        for head in heads:
            # find nearest head
            (nn_idx, nn_dst) = head.find_nearest(streamer.heads.pos)
            nn_head = streamer.heads[int(nn_idx)]

            # get the relative capacitance, which is ok
            c_nn = self.get_cap_factor(SHList([nn_head]))
            c_h = self.get_cap_factor(SHList([head]))
            # u' = q' / c' = u * c / c'
            c_frac = c_nn / c_h
            # ensure that the potential does not increase
            head.U0 = nn_head.U0 * min(1, c_frac)

            msg = 'Propagating head set to {:.1f} kV'
            logger.log(5, msg.format(head.U0 * 1e-3))
            if c_frac > 1:
                msg = 'Propagating potential capped.'
                logger.log(5, msg)

    def share_charge(self, streamer, heads):
        ''' Set potential of each given head and the closest existing head,
            by sharing charge between them.
        '''
        # Note: this routine may change the voltage of the needle!
        for head in heads:
            # find (the) nearest head(s)
            (nn_idx, nn_dst) = head.find_nearest(streamer.heads.pos)
            nn_head = streamer.heads[int(nn_idx)]
            shl = SHList([nn_head])  # should work for several neighbors also
            # shl = SHList(streamer.heads)  # to test with all heads

            # find total charge
            k = shl.calc_scale_nnls()
            c = self.get_cap_factor(shl)
            u = shl.U0
            q_tot = sum(ki * ci * ui for ki, ci, ui in zip(k, c, u))

            # append the new head and find scale
            shl.append(head)
            shl.k = [1 for _ in shl]
            shl.U0 = [1 for _ in shl]
            k = shl.calc_scale_nnls()

            # calculate the new individual potentials
            c = self.get_cap_factor(shl)
            ci_ki = sum(ki * ci for ki, ci in zip(k, c))
            v = [ki * q_tot / ci_ki for ki in k]
            shl.U0 = v

            # set the new (shared) potential
            epot = shl.epot(shl.pos)

            if epot.max() > max(u):
                epot[:] = max(u)
                msg = 'Warning! Charge sharing increasing potential prevented.'
                logger.warning(msg)

            if not np.allclose(epot[0], epot):
                logger.warning('Warning! Charge sharing issue!')
                diff = epot.max() / epot.min() - 1
                msg = 'Maximum relative difference, {:2.0f} %'
                logger.debug(msg.format(diff * 100))
                logger.log(5, 'Potentials {}'.format(epot))

            shl.U0 = epot  # note: they should all be equal

            msg = 'Branching heads set to {:.1f} kV'
            logger.log(5, msg.format(epot[0] * 1e-3))

    def get_final_potential(self, heads):
        # final potential (needle minus field in channel)
        length = self.origin.dist_to(heads.pos)
        length = length + eps  # prevents errors for needle
        return self.origin.U0 - length * self.U_grad

#
