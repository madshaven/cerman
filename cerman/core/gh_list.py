#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Defining a list of geometrical hyperboloids.

    Properties of the hyperboloids can be retrieved as lists.
    Methods for the hyperboloids are available for the lists.

    The class defines a nu-matrix which is defined as:
    M^(nu)_ij is the nu value at the position of gh i, given by gh j.

    The class also contains methods for choosing gh's to remove,
    based on given criteria.
'''

# General imports
import numpy as np
import logging

# Import from project files
from .geometrical_hyperboloid import GeometricalHyperboloid
from .coordinate_functions import find_nearest
from .coordinate_functions import sort_nearest
from .coordinate_functions import safe_hstack

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       DEFINE HYPERBOLOID LIST                                       #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class GHList(list):

    _item_class = GeometricalHyperboloid

    def _verify(self):
        for item in self:
            assert type(item) == self._item_class

    def to_dict_list(self):
        ''' Return a list of dicts of the parameters of the gh's. '''
        return [gh.to_dict() for gh in self]

    @classmethod
    def from_dict_list(cls, dict_list):
        ''' Return a list of correct class, based on input. '''
        return cls([cls._item_class.from_dict(d) for d in dict_list])

    @classmethod
    def from_lists(cls, pos, rp=None):
        ''' Return a GHList based on a list of positions. '''
        assert pos.shape[0] == 3    # assure correct input
        no = pos.shape[1]           # length of output
        if type(rp) is list:        # correct rp, if needed
            assert len(rp) == no
        else:
            rp = [rp] * no

        dict_list = [dict(pos=pos[:, i], rp=rp[i])
                     for i in range(no)]
        return cls.from_dict_list(dict_list)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Properties                                                #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    @property
    def pos(self):
        pos = [gh.pos for gh in self]  # list positions
        return safe_hstack(pos)  # works for empty list

    rp = property(lambda self: np.array([gh.rp for gh in self]))
    d = property(lambda self: np.array([gh.d for gh in self]))
    a = property(lambda self: np.array([gh.a for gh in self]))
    r = property(lambda self: np.array([gh.r for gh in self]))
    nu0 = property(lambda self: np.array([gh.nu0 for gh in self]))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Methods using positions                                   #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def is_inside(self, pos, nu0=None):
        ''' Return True where the position inside an hyperbole.

        Parameters
        ----------
        pos :       array(3, -1)
                    Cartesian positions
        nu0 :       float or array
                    defaults to nu0 of each hyperboloid

        Returns
        -------
        array[bool]
                True for positions within where (pos.nu0 < nu0)
        '''

        assert pos.shape[0] == 3, 'pos.shape[0] =! 3'

        out = np.zeros(pos.shape[1], dtype=bool)
        for gh in self:
            out |= gh.is_inside(pos, nu0=nu0)

        return out

    def find_nearest(self, pos_j=None, no=None):
        ''' Find the pos_j which is closest to each hyperbole.'''
        return find_nearest(self.pos, pos_j=pos_j, no=no)

    def dist_to(self, pos):
        ''' Find the distance from the tip of each hyperbole to each position.
        '''
        return np.linalg.norm(self.pos - pos, axis=0)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       NU/U MATRIX                                               #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def calc_M_nu(self):
        ''' Calculate the nu matrix, M_nu.

        Notes
        -----
        | M^(nu)_ij = nu(pos_i, head_j)
        | M^(nu)_ij is the nu value at the position of gh i, given by gh j.
        | M^(nu)_jj is equal nu0 for gh j.

        Head i is at a lower potential than gh j if:
        M^(nu)_ij > M^(nu)_jj

        Returns
        -------
        M_nu :  mat
                nu matrix
        '''
        pos = safe_hstack([gh.pos for gh in self])

        l = pos.shape[1]
        M_nu = np.zeros((l, l))

        for j in range(l):
            pos_ps = self[j].cart2ps(pos, to_calc='nu')
            M_nu[:, j] = pos_ps[1, :]

        # Assert values along diagonal
        assert np.allclose(np.diagonal(M_nu),
                           np.array([gh.nu0 for gh in self])
                           ),  'Assert values along diagonal'
        # Valid nu
        assert (M_nu >= 0).all(), '(M_nu >= 0).all()'
        assert (M_nu <= np.pi / 2).all(), '(M_nu <= np.pi/2).all()'

        return M_nu

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       TO REMOVE                                                 #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def to_remove_nu_mat(self, M_nu=None):
        ''' Find hyperboloids that are located within another.

        Position i is inside head j if:
        M^(nu)_ij < M^(nu)_jj

        Parameters
        ----------
        M_nu :  mat[float]
                nu matrix

        Returns
        -------
        GHList
        '''

        # works for empty list as well
        if len(self) < 2:
            ghl = []

        else:
            if M_nu is None:
                M_nu = self.calc_M_nu()

            # where returns a tuple of i and j positions
            # where the condition is met.
            i, j = np.where(M_nu < np.diagonal(M_nu))
            i = list(set(i))  # remove duplicates

            ghl = ([self[_i] for _i in i])

        return type(self)(ghl)  # works for derived classes

    def to_remove_dist_simple(self, dist):
        '''Find hyperboloids that are close to others and behind them.

        Info
        ----
        if |r_ij| < dist, remove the one with highest z-value.

        Parameters
        ----------
        dist :  float
                distance to consider close

        Returns
        -------
        GHList
        '''

        # Calculate nearest neighbour matrices
        M_idx, M_dst = sort_nearest(self.pos)

        ghl = set()  # create a list to populate
        for (i, gh) in enumerate(self):
            # Remove the gh with highest z-value if two are close
            # idx 0 is self, 1 is closest
            nn_idx = M_idx[i, 1]
            nn_dst = M_dst[i, nn_idx]
            nn_eh = self[nn_idx]
            if (nn_dst < dist):
                if (gh.z > nn_eh.z):
                    ghl.add(gh)
                else:
                    ghl.add(nn_eh)

        return type(self)(ghl)  # works for derived classes

    def to_remove_dist_recursive(self, dist):
        '''Find hyperboloids that are close to others and behind them.

        Use recursive/sorted procedure to:
        Find lowest --> remove everything close --> repeat

        Parameters
        ----------
        dist :  float
                distance to consider close

        Returns
        -------
        GHList
        '''

        M_idx, M_dst = sort_nearest(self.pos)

        # use sets to get every instance only once
        ghl_to_keep = set()
        ghl_to_remove = set()

        # sort by z, high to low
        ghl_sorted = sorted(self, key=lambda gh: -gh.z)

        # continue until none are left
        while ghl_sorted:
            gh_lowest = ghl_sorted.pop()      # gh with lowest z
            ghl_to_keep.add(gh_lowest)        # keep gh_lowest
            i = self.index(gh_lowest)         # get index

            # remove any gh that are close to gh_lowest
            ghl_to_remove |= set(
                gh
                for (j, gh) in enumerate(self)
                if (M_dst[i, j] < dist and i != j)
                )

            # remove removed gh from list
            ghl_sorted = [gh for gh in ghl_sorted if gh not in ghl_to_remove]

        return type(self)(ghl_to_remove)  # works for derived classes


#
