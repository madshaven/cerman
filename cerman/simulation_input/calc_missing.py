#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Calculate simulation parameters from other parameters.
'''

from ..simulate.calc_alpha import CalcAlpha
from .. import tools
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       SET/UPDATE PARAMETERS                                         #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def calc_missing_params(data_in):
    ''' Add defaults to user input and calculate derived parameters.

    Run this function prior to any simulation to ensure
    that the latest defaults and revision is captured.


    Parameters
    ----------
    data_in   : dict
                simulation input parameters

    Returns
    -------
    data_out :  OrderedDict
                Derived parameters
    '''

    # work on a copy (note that this is a shallow copy!)
    data_out = data_in.copy()

    if data_out['additive_cn'] is None:
        # todo: add testing for cw, cm, and cn
        # Only one should be given, and they are given priority in that order
        data_out['additive_cn'] = calc_cn(
            cw    = data_out['additive_cw'],
            rho_a = data_out['additive_density'],
            rho_l = data_out['liquid_density'],
            m_l   = data_out['liquid_mass'],
            m_a   = data_out['additive_mass'],
            )

    if data_out['seeds_cn'] is None:
        data_out['seeds_cn'] = calc_seeds_cn(
            sigma  = data_out['liquid_sigma'],
            e      = -1.6e-19,
            mu_ion = data_out['liquid_mu_ion'],
            )

    if data_out['liquid_Ec_ava'] is None:
        calc_alpha = CalcAlpha(
            kind = data_out.get('alphakind', None),
            ea   = data_out.get('liquid_Ealpha', None),
            am   = data_out.get('liquid_alphamax', None),
            ka   = data_out.get('liquid_k1', None),
            il   = data_out.get('liquid_IP', None),
            ia   = data_out.get('additive_IP', None),
            cn   = data_out.get('additive_cn', None),
            )
        data_out['liquid_Ec_ava'] = calc_alpha.ec(
            data_out['liquid_alphacrit'])

    if data_out['streamer_photo_enabled'] is True:
        if data_out['streamer_photo_efield'] is None:
            data_out['streamer_photo_efield'] = calc_photo_efield(
                IP = data_out['liquid_IP'],
                e1 = data_out['liquid_e1'],
                beta_b = data_out['liquid_beta_b'],
                R_perm = data_out['liquid_R_perm'],
                )
        if data_out['streamer_photo_efield'] is None:
            logger.error('Insufficient information for photoionization.')
            data_out['streamer_photo_enabled'] = False

    # Add revision information
    data_out['git_revision_local'] = tools.git_revision()
    data_out['git_revision_package'] = tools.version['git_revision']
    data_out['version'] = tools.version['version']
    data_out['full_version'] = tools.version['full_version']

    # return only new keys
    return OrderedDict(((key, val)
                        for key, val in data_out.items()
                        if data_in[key] is None
                        ))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       TOOLS                                                         #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Assuming; volume fraction = C / C_0, of a given substance,
# then, the number fraction is
# f_n = 1 / (1 + C_l0 (1 / C_a - 1 / C_a0))
# where l is liquid and a is additive.

def calc_cn(cw, rho_l, rho_a, m_l, m_a):
    ''' Calculate the mol fraction.

    Parameters
    ----------
    cw    : float
            [%] weight fraction
    rho_l : float
            liquid density
    rho_a : float
            additive density
    m_l   : float
            liquid molar mass
    m_a   : float
            additive molar mass

    Returns
    -------
    cn    : float
            mol fraction
    '''
    return (cw/100) * (rho_a/rho_l) / (m_a/m_l)


def fw2fn(fw, ml, ma):
    ''' Calculate number fraction from weight fraction.

    Note! Use weight fraction, not weight percent.

    Parameters
    ----------
    fw    : float
            weight fraction
    ml    : float
            liquid molar mass [mol/g]
    ma    : float
            additive molar mass [mol/g]

    Returns
    -------
    fn    : float
            mol fraction
    '''

    # weight fraction = mass_a / (mass_a + mass_l)
    # number fraction = mol_a / (mol_a + mol_l)
    # the expression below is reduced to
    # fw * ma / ml, when fw << 1

    return 1 / (1 + ml / ma * (1 / fw - 1))


def ca2fn(ca, rl, ml, ra, ma):
    ''' Calculate number fraction from concentration.

    Note! Assuming volume fraction given by concentration.

    Parameters
    ----------
    ca    : float
            additive concentration [M]
    rl    : float
            liquid density [kg/L]
    ml    : float
            liquid molar mass [mol/g]
    ra    : float
            additive density [kg/L]
    ma    : float
            additive molar mass [mol/g]

    Returns
    -------
    fn    : float
            mol fraction
    '''

    # given a concentration c, and max concentration c0,
    # assume; volume fraction = c / c0
    # the expression below is reduced to
    # ca / cl0, when ca << ca0 and ca << cl0

    cl0 = rl / ml * 1000  # [M]
    ca0 = ra / ma * 1000  # [M]
    return 1 / (1 + cl0 * (1 / ca - 1 / ca0))


def calc_seeds_cn(sigma, e, mu_ion):
    ''' Calculate the number concentration of seeds.

    Parameters
    ----------
    sigma  : flaot
             conductivity
    e      : float
             electron charge
    mu_ion : float
             mobility

    Returns
    -------
    float
        number concentration
    '''
    return 1/2 * sigma / (e * mu_ion)


def calc_photo_efield(IP, e1, beta_b, R_perm):
    ''' Return the electric field threshold for photoionization.

    Parameters
    ----------
    IP :    float
        ionization potential of liquid
    e1 :    float
        energy of radiation from excited state
    beata_b :   float
        IP field reduction parameter
    R_perm :    float
        relative permittivity

    Returns
    -------
    float
        threshold field for photoionization
    '''
    if (IP is None) or (e1 is None) or (beta_b is None) or (R_perm is None):
        logger.info(f'Could not calculate estr for photoionization')
        return None

    E_a = 5.14e11  # V/m
    return  (IP - e1)**2 / beta_b**2 * R_perm * E_a


#
