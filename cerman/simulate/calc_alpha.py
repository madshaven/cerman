#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Calculate the average ionization coefficient, alpha.
'''

# General imports
import logging
import numpy as np
from scipy.optimize import newton
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.optimize import fsolve

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class CalcAlpha():
    '''Wrapper to get correct function.'''

    def __new__(self, kind=None, **kwargs):
        if kind is None:
            cls = CalcAlphaGas
        elif kind == 'gas':
            cls = CalcAlphaGas
        elif kind == 'I2009':
            cls = CalcAlphaI2009
        elif kind == 'A1991':
            cls = CalcAlphaA1991
        else:
            logger.error(f'Could not return correct class! "{kind}"')
        return cls(**kwargs)


class CalcAlphaGas():
    '''Calculate alpha by gas approximation.'''

    def __init__(self, am, ea, **kwargs):
        self.am = am  # alpha max
        self.ea = ea  # characteristic field strength
        logger.debug('Initiated CalcAlpha')
        logger.debug('Ignored kwargs: ' + str(kwargs))

    def __call__(self, estr):
        # return corresponding alpha
        return self.am * np.exp(-self.ea / estr)

    def ec(self, ac):
        # return corresponding electric field strength
        x0 = self.ea * 0.1
        xtol = 1e-2
        maxfev = int(1e4)
        return fsolve(lambda x: self(x) - ac, x0=x0, xtol=xtol, maxfev=maxfev)[0]


class CalcAlphaI2009(CalcAlphaGas):
    '''Calculate alpha with additives by Ingebrigtsen approximation.

    alpha' = (1 - cn) * alpha + cn * alpha * exp(ka (il - ia))
    alpha' = alpha (1 - cn + cn (exp(ka (il - ia) )
    alpha' = alpha * additive_factor

    Note: Ingebrigtsen calculated ka based on molar concentration,
    instead of volume fraction.
    '''

    def __init__(self, am, ea, il, ia, ka, cn, **kwargs):
        # am : alpha max
        # ea : characteristic field strength
        # il : liquid ionization potential
        # ia : additive ionization potential
        # ka : factor derived by Ingebrigtsen
        # cn : additive mole fraction
        di = il - ia  # ionization potential difference
        af = (1 - cn) + cn * np.exp(ka * di)  # additive factor
        am = am * af
        super().__init__(am=am, ea=ea)

        logger.debug('Ignored kwargs: ' + str(kwargs))


class CalcAlphaA1991(CalcAlphaGas):
    '''Calculate alpha by Atrazhev approximation.

    eta = alpha / E = 3e/I * (Ev/E)**2 * exp(-(Ev/E)**2)
    alpha = 3e/I * (Ev/E)**2 * exp(-(Ev/E)**2) * E
    '''

    def __init__(self, ea, il, **kwargs):
        self.ea = ea  # characteristic field strength
        self.il = il  # ionization potential
        logger.debug('Initiated CalcAlpha')
        logger.debug('Ignored kwargs: ' + str(kwargs))

    def __call__(self, estr):
        # return corresponding alpha
        f = (self.ea / estr)**2
        return 3 / self.il * f * estr * np.exp(-f)


if __name__ == '__main__':
    # test alpha gas
    ca = CalcAlphaGas(am=200e-6, ea=3e9)  # haidara1991
    e = [1e8, 1e9, 1e10]
    a = [ca(ei) for ei in e]
    _e = [ca.ec(ai) for ai in a]
    print('test alpha gas:')
    print('E_init', ['{:.2e}'.format(ei) for ei in e])
    print('E_crit', ['{:.2e}'.format(ei) for ei in _e])
    print('Alphas', ['{:.2e}'.format(ai) for ai in a])

    # test alpha A1991
    ca = CalcAlphaA1991(ea=3e9, il=10)  # haidara1991
    e = [1e8, 1e9, 1e10]
    a = [ca(ei) for ei in e]
    _e = [ca.ec(ai) for ai in a]
    print('test alpha A1991:')
    print('E_init', ['{:.2e}'.format(ei) for ei in e])
    print('E_crit', ['{:.2e}'.format(ei) for ei in _e])
    print('Alphas', ['{:.2e}'.format(ai) for ai in a])
    #


#
