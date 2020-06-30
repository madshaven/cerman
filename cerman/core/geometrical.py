#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This module contains functions to convert between
Cartesian and prolate spheroid coordinates.
Prolate spheroid coordinates defines intersecting
prolate spheroids and hyperboloids.


Prolate spheroid coordinates
----------------------------
| x = a sinh mu cosh nu cos phi
| y = a sinh mu cosh nu sin phi
| z = a cosh mu sinh nu
|
| 2 a cosh mu = sqrt(x^2 + y^2 +(z+a)^2) + sqrt(x^2 + y^2 +(z+a)^2)
| 2 a cos  nu = sqrt(x^2 + y^2 +(z+a)^2) - sqrt(x^2 + y^2 +(z+a)^2)
| tan phi     = y/x
|
| z(x, y, mu) = cotanh mu sqrt(a^2 sinh^2 mu - x^2 - y^2)
| z(x, y, nu) = cotan  nu sqrt(a^2 sin ^2 mu + x^2 + y^2)
|
| Constant mu forms prolate spheroids
| z^2 / a^2 cosh^2 mu + (x^2 + y^2) / a^2 sinh^2 mu = cos ^2 nu + sin ^2 = 1
|
| Constant nu forms hyperboloids
| z^2 / a^2 cos ^2 nu - (x^2 + y^2) / a^2 sin ^2 nu = cosh^2 nu - sinh^2 = 1
|
| A hyperbola is given by z'^2 = x'^2 + 1,
| or more general (z/a')^2 = (x/b')^2 + 1.
|
| a' = a cos nu,  minimum z value
| b' = a sin nu,  distance to the focus
| e  = 1/cos nu,  eccentricity
|
| 0 <= e <= 1 for an ellipse
| e  = 1 for a parabola
| 1 <= e      for a hyperbola

'''

# General imports
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ooo = np.array([0, 0, 0]).reshape(3, -1)
iio = np.array([1, 1, 0]).reshape(3, -1)
iii = np.array([1, 1, 1]).reshape(3, -1)
ooi = np.array([0, 0, 1]).reshape(3, -1)
eps = np.finfo(float).eps  # 2.22e-16 for double


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       PROLATE TO CART                                               #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def ps2cart(pos_ps, ps_a):
    ''' Calculates the Cartesian position.

    Parameters
    ----------
    pos_ps  : array
              prolate spheroid coordinate(s)
    ps_a    : float
              hyperbola parameter

    Returns
    -------
    array
              Cartesian positions
    '''

    assert pos_ps.shape[0] == 3 and pos_ps.ndim == 2, (
        'Wong size. Shape {}.'.format(pos_ps.shape))

    pos_cart = np.zeros_like(pos_ps)

    sinh_sin = np.sinh(pos_ps[0]) * np.sin(pos_ps[1])
    pos_cart[0] = ps_a * sinh_sin * np.cos(pos_ps[2])     # x-coordinate
    pos_cart[1] = ps_a * sinh_sin * np.sin(pos_ps[2])     # y-coordinate
    pos_cart[2] = ps_a * np.cosh(pos_ps[0]) * np.cos(pos_ps[1])  # z-coordinate
    pos_cart[np.isnan(pos_cart)] = 0
    return pos_cart


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       CART TO PROLATE                                               #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def cart2ps(pos_cart, ps_a, to_calc=['mu', 'nu', 'phi']):
    ''' Calculates the prolate spheroid position.

    Parameters
    ----------
    pos_cart  : array
                Cartesian coordinate(s)
    ps_a      : float

    Return
    ------
    array
                prolate spheroid coordinate(s)
    '''
    assert pos_cart.shape[0] == 3, 'pos_cart.shape[0] =! 3'
    assert pos_cart.ndim <= 2, 'pos_cart.ndim[0] >= 2'
    assert ps_a > 0, 'Error: Negative value in ps_a.'

    pos_cart = np.array(pos_cart, copy=False).reshape(3, -1)
    pos_ps = np.zeros(pos_cart.shape)
    x = pos_cart[0, :]
    y = pos_cart[1, :]
    z = pos_cart[2, :]

    if ('mu' in to_calc) or ('nu' in to_calc):
        zp2 = (z + ps_a)**2
        zm2 = (z - ps_a)**2
        r2 = x**2 + y**2

        p_2a = np.sqrt(r2 + zp2) / (2 * ps_a)
        m_2a = np.sqrt(r2 + zm2) / (2 * ps_a)

    if 'mu' in to_calc:     # mu-coordinate
        cosh_mu = p_2a + m_2a
        # check for errors at z-axis, for |z| < a, (p + m = 2a)
        if np.any(cosh_mu < 1):
            mask = cosh_mu < 1
            msg = 'Found {} point(s) at axis, z < a.'
            logger.debug(msg.format(mask.sum()))
            # correct mu to zero at axis
            cosh_mu[mask] = 1

        pos_ps[0, :] = np.arccosh(cosh_mu)

    if 'nu' in to_calc:     # nu-coordinate
        cos_nu = p_2a - m_2a
        # check for errors at z-axis, for |z| > a, (p - m > 2a)
        if np.any(cos_nu > 1):
            mask = cos_nu > 1
            msg = 'Warning! Found {} point(s) at axis, z > a.'
            logger.warning(msg.format(mask.sum()))
            # "correct" nu to zero at axis
            cos_nu[mask] = 1

        pos_ps[1, :] = np.arccos (cos_nu)

    if 'phi' in to_calc:    # phi-coordinate
        pos_ps[2, :] = np.arctan2(y, x)

    # Add test to assure that this is required.
    # pos_ps[np.isnan(pos_ps)] = 0

    return pos_ps


#
