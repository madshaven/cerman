#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Defining a list of electrical hyperboloids.

    Properties of the hyperboloids can be retrieved as lists.
    Methods for the hyperboloids are available for the lists.

    The class defines a u-matrix, a U0-matrix
    which relates the potential at the tip of each hyperboloid
    to each other hyperboloid.
    The matrix can be used to remove eh's.

    The class also contains methods for scaling the potentials.
'''

# general imports
import numpy as np
import logging
import scipy.optimize

# import from project files
from .gh_list import GHList
from .electrical_hyperboloid import ElectricalHyperboloid
from . import coordinate_functions

# definitions
ooi = np.array([0, 0, 1]).reshape(3, -1)
eps = np.finfo(float).eps  # 2.22e-16 for double

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE ELECTRICALHYPERBOLOID LIST                             #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class EHList(GHList):

    _item_class = ElectricalHyperboloid

    @classmethod
    def from_lists(cls, pos, rp=None, k=None, U0=None, dtype=None):
        ''' Return an EHList based on one or more lists.
        '''
        assert pos.shape[0] == 3, 'pos size issue'    # assure correct input
        no = pos.shape[1]           # length of output

        # define a function to verify and make lists
        def _to_list(var):
            if type(var) is list:
                assert len(var) == no, 'wrong var length (list)'
            elif type(var) is np.ndarray:
                assert len(var) == no, 'wrong var length (array)'
            else:
                var = [var] * no
            return var

        # verify and listify the input parameters
        rp = _to_list(rp)
        k = _to_list(k)
        U0 = _to_list(U0)
        dtype = _to_list(dtype)

        # create a list of dicts
        dict_list = [
            dict(pos=pos[:, i], rp=rp[i], k=k[i], U0=U0[i], dtype=dtype[i])
            for i in range(no)
            ]

        # note GHList use _item_class to return correct class
        return cls.from_dict_list(dict_list)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Properties                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def U0(self):
        return np.array([eh.U0 for eh in self])

    @U0.setter
    def U0(self, U0):
        if isinstance(U0, (int, float)):
            U0 = [U0 for _ in self]
        assert len(self) == len(U0), 'len(self) != len(U0)'
        for eh, _U0 in zip(self, U0):
            eh.U0 = _U0

    @property
    def k(self):
        return np.array([eh.k for eh in self])

    @k.setter
    def k(self, k):
        if isinstance(k, (int, float)):
            k = [k for _ in self]
        assert len(self) == len(k), 'len(self) != len(k)'
        for eh, _k in zip(self, k):
            eh.k = _k

    U = property(lambda self: np.array([eh.U for eh in self]))
    e = property(lambda self: np.array([eh.e for eh in self]))
    c = property(lambda self: np.array([eh.c for eh in self]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Methods using positions                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # note: pos=None does not make sense here
    # note: see ElectricalHyperboloid for description of functions

    def evec(self, pos, dtype=None, is_inside=None):
        '''Calculate electric field vector.'''
        evec = np.zeros(pos.shape, dtype=dtype)
        for eh in self:
            # note: evec is modified in-place
            eh.evec(pos=pos, dtype=dtype, is_inside=is_inside, evec=evec)
        return evec

    def estr(self, pos, evec=None, dtype=None):
        '''Calculate electric field strength.'''
        if evec is None:
            evec = self.evec(pos=pos, dtype=dtype)
        estr = np.linalg.norm(evec, axis=0)
        return estr

    def edir(self, pos, evec=None, estr=None, dtype=None):
        '''Calculate electric field direction.'''
        if evec is None:
            evec = self.evec(pos=pos, dtype=dtype)
        if estr is None:
            estr = np.linalg.norm(evec, axis=0)
        return evec / (estr + eps)

    def epot(self, pos, dtype=np.float64):
        '''Calculate electric potential.'''
        epot = np.zeros(pos.shape[1], dtype=dtype)
        for eh in self:
            # note: epot is modified (updated/changed) in-place
            eh.epot(pos=pos, dtype=dtype, epot=epot)
        return epot

    def epot_integration(self, pos, pt_no=1e3, log=True):
        '''Calculate electric potential by integration of field.'''
        epot = np.zeros(pos.shape[1])
        for eh in self:
            epot += eh.epot_integration(pos=pos, pt_no=pt_no, log=log)
        return epot

    def eprop(self, pos, dtype=None):
        '''Calculate electric field properties.'''
        is_inside = np.zeros(pos.shape[1], dtype=bool)
        evec = self.evec(pos, dtype=dtype, is_inside=is_inside)
        estr = self.estr(pos, evec=evec)
        edir = self.edir(pos, evec=evec, estr=estr)
        return edir, estr, evec, is_inside

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       NU/U MATRIX                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def calc_M_u(self, M_nu=None):
        ''' Calculate the u-matrix, M_u.
        This is a geometrical factor relating
        the tip of one hyperbole to another hyperbole.

        Parameters
        ----------
        M_nu :  mat
                nu matrix

        Returns
        -------
        mat[float]
                u matrix

        Notes
        -----
        | M^(u)_ij = u_i / u_j
        | U = C ln cot nu/2
        | u = U/U0 = ln tan (nu / 2) / ln tan (nu0 / 2)
        | M^(u)_ij = ln tan (M^(nu)_ij / 2) / ln tan (M^(nu)_jj / 2)
        |          = U_ij / U_jj
        |
        | Given U_j = U0_j k_j as the voltage of eh j,
        | M^(u)_ij U_j = U^*_i
        | where U^*_i is the potential at position i, due to all the self.
        |
        | Head i is at a lower potential than eh j if:
        | M^(u)_ij < 1
        '''

        if M_nu is None:
            M_nu = self.calc_M_nu()

        assert M_nu.size == len(self)**2, 'M_nu.size != len(self)**2'

        # NOTE: diagonal returns a view!
        # Use a/=diag(a).copy() or a = a/diag(a)
        M_u = -np.log(np.tan(M_nu / 2))
        M_u /= np.diagonal(M_u).copy()
        return M_u

    def calc_M_U0(self, M_u=None, M_nu=None):
        ''' Calculate the U0-matrix, M_U0.
        This is the potential at the tip of each hyperbole
        given each other hyperbole.

        Parameters
        ----------
        M_u  :  mat[float]
                u matrix
        M_nu :  mat[float]
                nu matrix

        Returns
        -------
        M_U0 :  mat[float]
                U0 matrix

        Notes
        -----
        M^(U0)_ij = M^(u)_ij U0_j
        '''

        if M_u is None:
            M_u = self.calc_M_u(M_nu=M_nu)

        assert M_u.size == len(self)**2, 'M_u.size != len(self)**2'

        return M_u * self.U0

    def calc_M_U(self, M_u=None, M_nu=None):
        ''' Calculate the U-matrix, M_U.
        This is the scaled potential at the tip of each hyperbole
        given each other hyperbole.

        Parameters
        ----------
        M_u  :  mat[float]
                u matrix
        M_nu :  mat[float]
                nu matrix

        Returns
        -------
        M_U :   mat[float]
                U matrix

        Notes
        -----
        M^(U)_ij = M^(u)_ij U_j
        '''

        if M_u is None:
            M_u = self.calc_M_u(M_nu=M_nu)

        assert M_u.size == len(self)**2, 'M_u.size != len(self)**2'

        U = np.array([eh.U() for eh in self])
        return M_u * U

    def calc_scale_nnls(self, update=False, M_U0=None, M_u=None, M_nu=None):
        '''Return k scaled by nnls - non-negative least squares.

        The problem becomes overdetermined if there are lot of eh.
        Remove by other methods to improve the result.

        Parameters
        ----------
        update : bool
                update k for each hyperbola if True
        M_U0  : mat[float]
                U0 matrix
        M_u  :  mat[float]
                u matrix
        M_nu :  mat[float]
                nu matrix

        Returns
        -------
        k     : int
                scale

        Notes
        -----
        .. math:: M^{(U0)}_{ij} k_j = U^*_i

        :math:`U_j` is the voltage of hyperboloid j.
        :math:`U^*_i` is the potential at position i, due to all the heads.

        The problem is optimized by
        .. math:: M k - U = 0
        '''
        # Under special circumstances,
        # this may scale leading hyperboloid to zero.

        # Calculate M_U0 if required
        if M_U0 is None:
            M_U0 = self.calc_M_U0(M_u=M_u, M_nu=M_nu)

        assert M_U0.size == len(self)**2, 'M_U0.size != len(self)**2'

        # Optimize the potential matrix
        k, r = scipy.optimize.nnls(M_U0, self.U0)

        # # Check for hyperboloid set to 0
        # if (k == 0).sum() > 0:
        #     msg = 'Problem with scaling potential. '
        #     msg += '{} hyperboloids(s) set to zero.'
        #     logger.info(msg.format((k == 0).sum()))

        # Check the residual. Proper value TBD
        if (r > 1e-2):
            logger.log(5, 'Problem with scaling potential. Large residual.')

        # update / set the scale for each hyperboloid
        if update:
            self.k = k

        return k

    def calc_scale_int(self, update=False, no=None, dist=None, debug=True):
        '''Return k scaled by integration of potentials of n-nearest neighbors.

        Parameters
        ----------
        no    : int
                number of closest hyperboloids to consider
        dist  : float
                consider only hyperboloids within this distance

        Returns
        -------
        k     : int
                hyperboloid scale
        '''
        # todo: check function! useful?

        l = len(self)
        pots_integ = np.zeros(l)
        pots_lapla = np.zeros(l)

        if (no is None) or (no > l):
            no = l

        assert no > 0, 'no == 0'

        k_prev = [eh.k for eh in self]

        # Reset scaling
        for eh in self:
            eh.k = 1

        M_i, M_d = coordinate_functions.sort_nearest(self.pos)

        # Set dist to some large number if it is not defined
        if dist is None:
            dist = M_d.max() * 2

        # Calculate potential from n-nearest self
        for i in range(l):
            # Pick out the 'no' nearest self that are closer than 'dist'.
            nn = M_i[i, :].tolist()
            nearest_heads = [self[n] for n in nn[:no] if M_d[i, n] < dist]
            # Calculate the potential
            pots_integ[i] = EHList(nearest_heads).epot_integration(self.pos[:, i])
            if debug:
                pots_lapla[i] = EHList(nearest_heads).epot(self.pos[:, i])

        if debug:
            msg = 'Potential diffs:\n U0_i/U0_l-1 {}'
            logger.debug(msg.format((pots_integ / pots_lapla - 1)))

        # Set scale
        k = np.zeros(l)
        for i, eh in enumerate(self):
            k[i] = eh.U0 / pots_integ[i]       # For reset eh

        if update:
            for eh, ki in zip(self, k):
                eh.k = ki
        else:
            for eh, ki in zip(self, k_prev):
                eh.k = ki

        if debug:
            pots_lapla = EHList(nearest_heads).epot(self.pos)

            msg = 'Potential diff after: self.U0/U0_l-1 {}'
            logger.debug(msg.format(self.U0 / pots_lapla - 1))

        return k

    def set_scale(self, mode=None):
        # scale the heads
        if mode == 'nnls':
            # assume that heads are correct and scale
            self.k = self.calc_scale_nnls()
        elif (mode == '1') or (mode == 1):
            self.k = 1
        elif (mode == 'none') or (mode is None):
            pass  # keep scaling
        else:
            logger.error(f'Error, invalid value for mode `{mode}`')
        return self.k

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       TO REMOVE                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def to_remove_u_mat(self, M_u=None, M_nu=None):
        ''' Find hyperboloids that are located within another.

        Position i is inside head j if:
        M^(u)_ij > 1
        M^(nu)_ij < M^(nu)_jj

        Parameters
        ----------
        M_u  :  mat[float]
                u matrix
        M_nu :  mat[float]
                nu matrix

        Returns
        -------
        EHList
        '''

        # works for empty list as well
        if len(self) < 2:
            ehl = []

        else:
            if M_u is None:
                M_u = self.calc_M_u(M_nu=M_nu)

            # where returns a tuple of i and j positions
            # where the condition is met.
            i, j = np.where(M_u > 1)
            i = list(set(i))  # remove duplicates

            ehl = ([self[_i] for _i in i])

        return type(self)(ehl)


#
