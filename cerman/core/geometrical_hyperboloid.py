#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' A hyperboloid class, based on prolate spheroid coordinates.

The hyperboloid is here defined by
the distance where it is closest to the plane (minimum z)
and its radius at the tip,
as well as its x and y positions.

The methods of the class compensates for its x and y positions
when e.g. converting between coordinate systems.
'''

# General imports
import numpy as np
import logging

# Import from project files
from .geometrical import cart2ps
from .geometrical import ps2cart
from .coordinate_functions import find_nearest

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ooo = np.array([0, 0, 0]).reshape(3, -1)
iio = np.array([1, 1, 0]).reshape(3, -1)
iii = np.array([1, 1, 1]).reshape(3, -1)

eps = np.finfo(float).eps  # 2.22e-16 for double


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE HYPERBOLOID                                            #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class GeometricalHyperboloid():
    '''Geometrical hyperboloid.

    Parameters
    ----------
    pos : array
          position, (3, 1)
    rp  : float
          tip radius

    Properties
    ----------
    d   : float
          distance to plane
    a   : float
          distance
    nu0 : float
          ps coordinate
    r   : float
          distance from z-axis
    '''

    def __init__(self, pos, rp=None):
        if rp is None:
            rp = eps    # sharp hyperboloid
        # Note the setters of `rp` and `pos` invokes `_update`
        self._rp = rp   # hack-set this one to prevent `_update`
        self.pos = pos  # set this one properly to `_update`

    def _update(self):
        # updates properties after pos or rp is changed

        # Coordinates
        self.x = self.pos[0]
        self.y = self.pos[1]
        self.z = self.pos[2]
        if self.z < 0:
            logger.warning(5, f'Hyperbole below zero {self.pos[2, 0]}')

        # Geometric
        self.d = np.asscalar(self.z)    # the minimum z-value of the hyperbole
        self.a = self.d + self.rp / 2   # the distance to the focus
        self.nu0 = np.arccos(self.d / self.a)  # asymptotic angle
        self.cos_nu0 = self.d / self.a
        self.r = np.asscalar(np.sqrt(self.x**2 + self.y**2))

    @property
    def pos(self):
        ''' The position of the tip of the hyperbole. '''
        return self._pos

    @pos.setter
    def pos(self, pos):
        pos = np.array(pos).reshape(3, 1)  # force error if wrong
        self._pos = pos
        self._update()  # the reason for having a getter and setter

    @property
    def rp(self):
        ''' The hyperbole tip radius. '''
        return self._rp

    @rp.setter
    def rp(self, rp):
        self._rp = np.asscalar(np.array(rp))  # ensure scalar
        self._update()  # the reason for having a getter and setter

    def __repr__(self):
        return f'GeometricalHyperboloid(pos={self.pos}, rp={self.rp})'

    def copy(self):
        ''' Return a copy of self. '''
        return eval(repr(self))

    # note: to_dict is mainly used for saving data
    #       it is easier to manage later than repr
    def to_dict(self):
        ''' Return a dict with the instance parameters. '''
        return {'rp': self.rp, 'pos': self.pos}

    @classmethod
    def from_dict(cls, d):
        ''' Return a class instance from a dictionary. '''
        return cls(**d)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Methods using positions                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def ps2cart(self, pos_ps, to_calc=['x', 'y', 'z']):
        ''' Return Cartesian position corresponding to `pos_ps`.
            The origin of the hyperbole is added to the output.

        Parameters
        ----------
        pos_ps :    array(3, -1)
                    prolate spheroid positions
        to_calc :   lst(char)
                    list of coordinates to calculate.
                    ['x', 'y', 'z'] is default

        Returns
        -------
        pos_cart :  array(3, -1)
                    Cartesian positions
        '''
        pos_ps = np.array(pos_ps).reshape(3, -1)
        pos_cart = np.zeros_like(pos_ps)

        if ('x' in to_calc) or ('y' in to_calc):
            sinh_sin = np.sinh(pos_ps[0]) * np.sin(pos_ps[1])
        if 'x' in to_calc:
            pos_cart[0] = self.a * sinh_sin * np.cos(pos_ps[2])
            pos_cart[0] += self.x
        if 'y' in to_calc:
            pos_cart[1] = self.a * sinh_sin * np.sin(pos_ps[2])
            pos_cart[1] += self.y
        if 'z' in to_calc:
            pos_cart[2] = self.a * np.cosh(pos_ps[0]) * np.cos(pos_ps[1])

        # todo: assure that this is required.
        pos_cart[np.isnan(pos_cart)] = 0
        return pos_cart

    def cart2ps(self, pos_in, to_calc=['mu', 'nu', 'phi']):
        ''' Return prolate spheroid position corresponding to `pos_in`
            relative to the position of the hyperbole.

        Parameters
        ----------
        pos_in :    array(3, -1)
                    Cartesian positions
        to_calc :   lst(char)
                    list of coordinates to calculate.
                    ['mu', 'nu', 'phi'] is default

        Returns
        -------
        pos_cart :  array(3, -1)
                    prolate spheroid positions
        '''

        pos_cart = pos_in - self.pos * iio  # change origin
        return cart2ps(pos_cart, ps_a=self.a, to_calc=to_calc)

    def is_inside(self, pos, nu0=None):
        ''' Return True where the position inside the hyperbole.

        Parameters
        ----------
        pos :       array(3, -1)
                    Cartesian positions
        nu0 :       float or array
                    defaults to nu0

        Returns
        -------
        array[bool]
                True for positions within where (pos.nu0 < nu0).
        '''
        assert pos.shape[0] == 3, 'pos.shape[0] =! 3'
        if nu0 is None:
            nu0 = self.nu0

        # pre-calculation
        # marginally faster, and much more readable
        # than inserting the expressions directly
        x2 = (pos[0, :] - self.x)**2
        y2 = (pos[1, :] - self.y)**2
        zp = (pos[2, :] + self.a)**2
        zm = (pos[2, :] - self.a)**2

        cos_nu_2a = np.sqrt(x2 + y2 + zp) - np.sqrt(x2 + y2 + zm)
        cos_nu0_2a = 2 * self.a * np.cos(nu0)
        # Note: for nu < nu_0, cos nu > cos nu_0.
        return cos_nu_2a > cos_nu0_2a

    def find_nearest(self, pos_j=None, no=None):
        ''' Find the pos_j which is closest to the tip of the hyperbole.'''
        return find_nearest(self.pos, pos_j=pos_j, no=no)

    def dist_to(self, pos):
        ''' Find the distance from the tip of the hyperbole to each position.
        '''
        return np.linalg.norm(self.pos - pos, axis=0)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Methods giving positions                                  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def trace_nu(self, nu0=None, xdir='xz', mu_lim=0.5, num=50):
        ''' Calculate the z values for a constant nu0 line
            in xz or yz direction.

        Parameters
        ----------
        nu0   : float
                self.nu0 is default
        xdir  : str
                plane to plot
        mu_lim : float
                mu limit
        num   : int
                number of points

        Returns
        -------
        array[float]
                x or y values
        array[float]
                z values
        '''

        assert xdir in ['xz', 'yz'], 'wrong xdir'

        if nu0 is None:
            nu0 = self.nu0

        # Create linspace and calculate values
        mu = np.linspace(-mu_lim, mu_lim, num=num)
        x = self.a * np.sinh(mu) * np.sin(nu0)
        z = self.a * np.cosh(mu) * np.cos(nu0)

        # Adjust for origin
        if xdir == 'xz':
            x += self.x
        elif xdir == 'yz':
            x += self.y

        # fixme:
        if __debug__ and False:
            if xdir == 'xz':
                poses = np.vstack((x, 0 * x, z))
            elif xdir == 'yz':
                poses = np.vstack((0 * x, x, z))

            pos_ps = self.cart2ps(poses)
            # Tolerance due to mu=0 returning mu=1e-8
            assert np.allclose(np.absolute(mu), pos_ps[0, :], atol=1e-7)
            # Assert all neighbors are equal
            assert np.allclose(pos_ps[1, :], np.roll(pos_ps[1, :], 1))

        return x, z

    def trace_nu_2D(self, nu=None, xdir='xz', offset=None, xlim=None, num=100):
        ''' Calculate the z values for a constant nu line
            in xz or yz direction.

        Parameters
        ----------
        nu  :   float
                self.nu0 is default
        xdir  : str
                plane to plot
        offset: float
            None: through self,
            0:    through origin
        x_lim : float
                x limit
        num   : int
                number of points

        Returns
        -------
        array[float]
                x or y values
        array[float]
                z values
        '''

        assert xdir in ['xz', 'yz'], 'wrong xdir'

        if nu is None:
            nu = self.nu0
        if xlim is None:
            xlim = self.rp * 20

        if offset is None:
            offset = 0
        else:
            if xdir == 'xz':
                offset -= self.y
            elif xdir == 'yz':
                offset -= self.x

        # Create linspace and calculate
        x = np.linspace(-xlim, xlim, num=num)

        r2 = x**2 + offset**2
        a2_snu2_inv = 1 / (self.a**2 * np.sin(nu)**2)
        z = self.a * np.cos(nu) * np.sqrt(1 + r2 * a2_snu2_inv)

        # Adjust for origin
        if xdir == 'xz':
            x += self.x
        elif xdir == 'yz':
            x += self.y

        debug = False
        if debug and __debug__:
            if xdir == 'xz':
                poses = np.vstack((x, 0 * x + offset, z))
            elif xdir == 'yz':
                poses = np.vstack((0 * x + offset, x, z))
            pos_ps = self.cart2ps(poses)
            if debug:
                print('trace_nu_2D')
                print(nu, offset)
                print(pos_ps[1, :])
                # print(pos_ps)
            assert np.allclose(pos_ps[1, :], nu)
            assert np.allclose(pos_ps[1, :], np.roll(pos_ps[1, :], 1))

        return x, z


#
