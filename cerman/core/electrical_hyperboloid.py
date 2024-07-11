#!/usr/bin/env python3
# -*- coding: utf-8 -*-


''' An electrical hyperboloid class.

The electrical hyperboloid is based on the geometrical hyperboloid.
The hyperboloid is a surface of constant potential, scaled by a given factor.

Prolate spheroid coordinates
----------------------------
| x = a sinh mu cosh nu cos phi
| y = a sinh mu cosh nu sin phi
| z = a cosh mu sinh nu

Electric properties
-------------------
| U(nu) = C ln cot nu/2
| C  = U0 / ln cot nu0/2
| E(mu, nu) = C / a sin nu sqrt(cosh^2 mu - cos^2 nu)
| E_hat = nu_hat
| nu_hat(x, y, z, nu) = [x, y, - z tan^2 nu] / (
|   a sin nu sqrt(sinh^2 mu + cosh^2 mu tan^2 nu)
| nu_hat \sim x x_hat + y y_hat - z tan^2 nu z_hat
|
| p2 = x**2 + y**2 + (z + a)**2
| m2 = x**2 + y**2 + (z - a)**2
| p' = sqrt(p2) / 2a
| m' = sqrt(m2) / 2a
| cos nu = p' - m'
|
| tan nu/2 = sin nu / (cos nu + 1) = sqrt(1 - cos nu / 1 + cos nu)
| lg tan nu/2 = 1/2 lg (1 - p' + m') - 1/2 lg (1 + p' - m')

'''

# general imports
import numpy as np
import logging

# import from project files
from .geometrical_hyperboloid import GeometricalHyperboloid
from . import coordinate_functions

# definitions
ooi = np.array([0, 0, 1]).reshape(3, -1)
eps = np.finfo(float).eps  # 2.22e-16 for double

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE ELECTRICALHYPERBOLOID                                  #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class ElectricalHyperboloid(GeometricalHyperboloid):
    '''Electrical hyperboloid.

    Parameters
    ----------
    U0  : float, optional
          Potential at eh
    k   : float, optional
          Potential scaling factor
    dtype : dtype
            data type to use for field calculations

    Properties
    ----------
    c   : float
          C/U0 = 1/ln cot (nu0/2)
    e0  : float
          E0/U0 = C/(rp*U0)
    '''

    def __init__(self, pos, rp, U0=1, k=1, dtype=np.float64):
        super().__init__(pos=pos, rp=rp)
        self._rp = rp   # hack-set this one to prevent `_update`
        self.pos = pos  # set this one properly to `_update`
        self.U0 = U0
        self.k = k
        self.dtype = dtype
        # note: changed everything from `asscalar` to `array`
        # consider adding k-threshold for field calculations

    def _update(self):
        # called after pos or rp is changed
        super()._update()

        # Electric properties
        self.c = -1 / np.log(np.tan(self.nu0 / 2))
        self.e0 = self.c / self.rp

    def __repr__(self):
        msg = 'ElectricalHyperboloid(pos={}, rp={}, U0={}, k={}, dtype={})'
        msg = msg.format(self.pos, self.rp, self.U0, self.k, self.dtype)
        return msg

    def to_dict(self):
        out = super().to_dict()
        return dict(out, U0=self.U0, k=self.k, dtype=self.dtype)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Properties                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def U0(self):
        ''' The potential along the surface of the hyperboloid. '''
        return self._U0

    @U0.setter
    def U0(self, U0):  # ensure scalar
        self._U0 = np.array(U0).item()

    @property
    def k(self):
        ''' The potential shielding coefficient.
        The potential is scaled by this factor.
        '''
        return self._k

    @k.setter
    def k(self, k):  # ensure scalar
        self._k = np.array(k).item()

    @property
    def U(self):
        '''Return the scaled potential at eh.'''
        return self.U0 * self.k

    @property
    def C(self):
        '''Return C constant for e-field and potential calculations.'''
        # C \approx 2 * V0/np.log(1+4*d/rp)
        # C = - V0 / np.log(np.tan(nu0/2))
        return self.U0 * self.k * self.c

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Methods using positions                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _estr(self, pos):
        ''' Calculate the Laplacian electric field strength.
            Can be used to verify field calculation approximation.
        '''
        # Transform coordinates
        pos_ps = self.cart2ps(pos, to_calc=['mu', 'nu'])

        # Calculate the strength of the electric field
        sin_nu = np.sin(pos_ps[1])
        cos_nu = np.cos(pos_ps[1])
        estr = (
            self.C() * np.ones(pos.shape[1]) / (
                self.a * sin_nu *
                np.sqrt(np.cosh(pos_ps[0])**2 - cos_nu**2)
                )
            )
        return estr

    ''' How the electric field is calculated.
    | Unit vectors, nu
    | nu_hat_i = 1/h_nu pd x_i nu
    | nu_hat_x:  cos(nu)*cos(phi)*sinh(mu)/sqrt(sin(nu)**2 + sinh(mu)**2)
    | nu_hat_y:  sin(phi)*cos(nu)*sinh(mu)/sqrt(sin(nu)**2 + sinh(mu)**2)
    | nu_hat_z:  -sin(nu)*cosh(mu)/sqrt(sin(nu)**2 + sinh(mu)**2)
    |
    | nu_hat_x (-m + p)*sqrt(-4*a**2 + m**2 + 2*m*p + p**2)*cos(phi)/
    |          (4*a*sqrt(m*p))
    | nu_hat_y (-m + p)*sqrt(-4*a**2 + (m + p)**2)*sin(phi)/
    |          (4*a*sqrt(m*p))
    | nu_hat_z (-m - p)*sqrt(4*a**2 - (p - m)**2)/
    |          (4*a*sqrt(m*p))
    |
    | E_str = C * 2 * a / (sqrt(m*p)*sqrt(4*a**2 - (p - m)**2))
    |
    | E_x = C * sqrt(-4*a**2 + (m + p)**2)*(-m + p)*cos(phi)/
    |       (2*m*p*sqrt(4*a**2 - (m - p)**2))
    | E_y = C * sqrt(-4*a**2 + (m + p)**2)*(-m + p)*sin(phi)/
    |       (2*m*p*sqrt(4*a**2 - (m - p)**2))
    | E_z = C * -(m + p)/(2*m*p)
    '''
    def evec(self, pos=None, dtype=None, evec=None, is_inside=None):
        ''' Calculate the Laplacian electric field.

        Note: if `evec` is given, the result is added inplace, then returned.
        Note: if `is_inside` is given, its result is added inplace only.

        Parameters
        ----------
        pos :   array (3, -1)
                positions
        dtype : dtype
                data precision of calculation and returned values
        evec :  array (3, -1)
                electric field vector,
                the result is added inplace to this array, if given
                the combined result is then returned
        is_inside :     array[bool] (3, -1)
                true for each position within the hyperboloid
                the result is added inplace to this array, if given

        Returns
        -------
        evec :  array (3, -1)
                electric field vector
        '''
        if evec is None:
            evec = np.zeros(pos.shape, dtype=dtype)
        assert evec.shape == pos.shape
        if pos is None:
            edir = - ooi
            estr = self.U0 * self.k * self.e0
            evec += estr * edir
            return evec.astype(dtype, copy=False)

        if dtype is None:
            dtype = self.dtype

        # eps is 2.22e-16 for double, 1.19e-7 for single
        eps = np.finfo(dtype).eps

        # transform coordinate system and correct dtype
        a = np.array(self.a).astype(dtype, copy=False)
        d = np.array(self.d).astype(dtype, copy=False)
        C = np.array(self.C).astype(dtype, copy=False)
        x = (pos[0, :] - self.x).astype(dtype, copy=False)
        y = (pos[1, :] - self.y).astype(dtype, copy=False)
        zp = (pos[2, :] + a).astype(dtype, copy=False)
        zm = (pos[2, :] - a).astype(dtype, copy=False)

        # intermediate calculations
        u2 = x**2 + y**2 + eps**2  # eps prevents error for u = 0
        p2 = u2 + zp**2
        m2 = u2 + zm**2
        p = np.sqrt(p2)    # 2a cosh mu = p + m
        m = np.sqrt(m2)    # 2a cos  nu = p - m
        u = np.sqrt(u2)
        sin_phi = y / u
        cos_phi = x / u
        s_mu = (-4 * a**2 + (p + m)**2)  # (2a sinh mu)**2
        s_nu = (+4 * a**2 - (p - m)**2)  # (2a sin  nu)**2

        def _correction(mask):
            # correct or mitigate errors caused by points on the axis
            s_mu[mask] = 0
            s_nu[mask] = 1

            # debug
            msg = '{} NaN set to zero in xy-direction.'
            logger.debug(msg.format(mask.sum()))

            # deep debug
            logger.log(5, 'Hyperboloid at')
            msg = 'a:{}, pos:{}'.format(self.a, self.pos.ravel())
            logger.log(5, msg)
            msg = 'NaN position(s) at'
            logger.log(5, msg)
            for pos_i in pos[:, mask].T:
                logger.log(5, '{}'.format(pos_i))
            logger.log(5, 's_mu {}'.format((s_mu)[mask]))
            logger.log(5, 's_nu {}'.format((s_nu)[mask]))

        # check for errors at z-axis, for |z| < a, (p + m = 2a)
        if np.any(s_mu <= 0):
            # correct floating point error by setting to zero in xy
            mask = (s_mu <= 0)
            msg = 'Found {} point(s) at axis, z < a.'
            logger.debug(msg.format(mask.sum()))
            _correction(mask)

        # check for errors at z-axis, for |z| > a, (p - m > 2a)
        if np.any(s_nu <= 0):
            # mitigate error by setting to zero in xy (not correct)
            mask = (s_nu <= 0)
            msg = 'Warning! Found {} point(s) at axis, z > a.'
            logger.warning(msg.format(mask.sum()))
            _correction(mask)

        # combining intermediate results
        C_2mp = C / (2 * m * p)
        C_2mp_xy = C_2mp * (p - m) * np.sqrt(s_mu / s_nu)
        # note: numpy sometimes gives a warning for sqrt(0/1)

        # calculating and adding electric field vector
        evec[0, :] += C_2mp_xy * cos_phi
        evec[1, :] += C_2mp_xy * sin_phi
        evec[2, :] += - C_2mp * (p + m)

        # check whether the position is located withing a eh
        # 2a cos nu > 2a cos nu0
        if is_inside is not None:
            is_inside |= ((p - m) > 2 * d)  # force an error if used wrongly

        return evec

    def estr(self, pos=None, evec=None, dtype=None):
        ''' Calculate the electric field strength
            as the norm of the electric field vector.
            Calculate evec if not given.
        '''
        if evec is None:
            evec = self.evec(self, pos=pos, dtype=dtype)
        return np.linalg.norm(evec, axis=0)

    def edir(self, pos=None, evec=None, estr=None, dtype=None):
        ''' Calculate the unit vector of the electric field.
            Calculate evec and estr if not given.
        '''
        if evec is None:
            evec = self.evec(self, pos=pos, dtype=dtype)
        if estr is None:
            estr = np.linalg.norm(evec, axis=0)
        return evec / (estr + eps)  # add eps for safe division

    def epot(self, pos=None, dtype=None, epot=None):
        '''Calculate the Laplacian electric potential.

        Note: if `epot` is given, the result is added inplace, then returned.

        Parameters
        ----------
        pos  :  array (3, -1)
                positions
        epot :  array (-1)
                electric potential, modified inplace if given

        Returns
        -------
        epot :  array (-1)
                electric potential

        Notes
        -----
        U = C ln cot nu/2
        C = U0 / ln cot nu/2
        '''

        if pos is None:  # return potential at surface
            return self.U0 * self.k

        if dtype is None:
            dtype = self.dtype

        # give error on wrong input
        assert pos.shape[0] == 3, 'pos.shape[0] =! 3''pos.shape[0] =! 3'

        # convert to correct dtype
        a = np.array(self.a).astype(dtype, copy=False)
        C = np.array(self.C).astype(dtype, copy=False)
        dx = (pos[0, :] - self.x).astype(dtype, copy=False)
        dy = (pos[1, :] - self.y).astype(dtype, copy=False)
        zp = (pos[2, :] + a).astype(dtype, copy=False)
        zm = (pos[2, :] - a).astype(dtype, copy=False)

        # calculate the potential at given precision
        u2 = dx**2 + dy**2
        p = np.sqrt(u2 + zp**2)
        m = np.sqrt(u2 + zm**2)

        if epot is None:
            epot = np.zeros(pos.shape[1], dtype=dtype)
        assert epot.shape[0] == pos.shape[1]

        # note: this is a bit strange work flow
        epot += - C * 0.5 * np.log((2 * a - p + m) / (2 * a + p - m))

        return epot

    def epot_integration(self, pos, pt_no=1e3, log=True):
        ''' Calculate the potential by integration of the electric field.

        Parameters
        ----------
        pos   : array
                positions
        pt_no : int
                number of integration points
        log   : bool
                True for log-int-points, False for linear

        Returns
        -------
        epot : array (-1)
                electric potential
        '''

        pt_no = int(pt_no)
        pos = np.array(pos).reshape(3, -1)

        epot = np.zeros(pos.shape[1])

        # for i in range(pos.shape[1]):
        for i in np.where(pos[2, :] > 0)[0]:  # np.where returns a tuple
            pos_i = pos[:, i].reshape(3, -1)
            intergr_pos = np.zeros((3, pt_no)) + pos_i
            if log:
                intergr_pos[2, :] -= np.logspace(
                    -12, np.log10(pos_i[2]), pt_no, endpoint=True)
            else:
                intergr_pos[2, :] -= np.linspace(
                    0, pos_i[2], pt_no, endpoint=True)

            evec = self.evec(intergr_pos)
            epot[i] = np.trapz(evec[2], intergr_pos[2])

        return epot

    def epot_pole(self, pos):
        ''' Calculate the electric potential from a monopole.
            Center is at d + rp.
            The potential is set to constant within the pole.
            Image charge is also considered.
        '''

        # get field at self.pos
        if pos is None:
            pos = self.pos

        # center of poles
        pos_pole = self.pos + ooi * self.rp
        pos_image = pos_pole - 2 * ooi * pos_pole

        # pot pole
        r_pole = pos - pos_pole
        rn_pole = np.linalg.norm(r_pole, axis=0)
        epot_pole = self.U * self.rp / rn_pole

        # pot image
        r_image = pos - pos_image
        rn_image = np.linalg.norm(r_image, axis=0)
        epot_image = self.U * self.rp / rn_image

        # restrict potential within the pole
        epot = epot_pole + epot_image
        epot[rn_pole < self.rp] = self.U
        # epot[rn_image < self.rp] = 0  # not needed

        return epot

    def evec_pole(self, pos=None, dtype=None):
        '''Pretend the hyperbole is a pole, and calculate the electrical field.

        Note: this is a dummy function, just used to see some results.

        Parameters
        ----------
        pos   : array (3, -1)
                positions

        Returns
        -------
        evec : array (3, -1)
                electric field vector
        '''

        # get field at self.pos
        if pos is None:
            pos = self.pos

        if dtype is None:
            dtype = self.dtype

        # eps is 2.22e-16 for double, 1.19e-7 for single
        eps = np.finfo(dtype).eps

        # center of poles
        pos_pole = self.pos + ooi * self.rp
        pos_image = pos_pole - 2 * ooi * pos_pole

        # evec pole
        r_pole = pos - pos_pole
        rn_pole = np.linalg.norm(r_pole, axis=0)
        evec_pole = self.U * self.rp * r_pole / rn_pole**3

        # evec image
        r_image = pos - pos_image
        rn_image = np.linalg.norm(r_image, axis=0)
        evec_image = self.U * self.rp * r_image / rn_image**3

        # set field to zero within the pole
        evec = evec_pole + evec_image
        evec[:, rn_pole < self.rp] = 0
        # evec[:, rn_image < self.rp] = 0

        return evec

    def estr_pole(self, pos=None, evec=None, dtype=None):
        ''' Pretend the hyperbole is a pole, calculate the electrical strength.
        '''
        if evec is None:
            evec = self.evec_pole(self, pos=pos, dtype=dtype)
        return np.linalg.norm(evec, axis=0)

    def edir_pole(self, pos=None, evec=None, estr=None, dtype=None):
        ''' Pretend the hyperbole is a pole,
            calculate the electrical field direction.
        '''
        if evec is None:
            evec = self.evec_pole(self, pos=pos, dtype=dtype)
        if estr is None:
            estr = np.linalg.norm(evec, axis=0)
        return evec / (estr + eps)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Class methods                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # todo: needed?
    @classmethod
    def epot_ps(cls, pos, d, rp, U0, dtype=np.float64):
        eh = cls(pos=np.array([0, 0, d]), rp=rp, U0=U0)
        return eh.epot(pos, dtype=dtype)


#
