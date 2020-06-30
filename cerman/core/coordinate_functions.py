#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Tools for positions (coordinates), creation, sorting, getting properties.

    Positions are arrays of shape (3, n),
    i.e.
    x = arr[0, :],
    y = arr[1, :],
    z = arr[2, :].
'''

# General imports
import numpy as np
import logging

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# definitions
ooo = np.array([0, 0, 0]).reshape(3, -1)
iio = np.array([1, 1, 0]).reshape(3, -1)
iii = np.array([1, 1, 1]).reshape(3, -1)

eps = np.finfo(float).eps  # 2.22e-16 for double



def safe_hstack(pos):
    # wrapper for numpy.hstack that works for empty lists
    return np.hstack([np.zeros((3, 0))] + pos)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       RANDOM                                                        #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def random_cyl_pos(r0, r1, z0, z1, no=1, cn=None):
    ''' Creates an array of uniformly placed positions
    within a cylinder of the given parameters.

    The number density "cn" is used if defined,
    else "no" is used.

    Parameters
    ----------
    r0 : float
         Minimum radius
    r1 : float
         Maximum radius
    z0 : float
         Lower z position
    z1 : float
         Higher z position
    no : float
         Number of positions
    cn : float
         Number concentration of positions

    Returns
    -------
    array(3, n)
         positions
    '''

    assert r0 >= 0, 'r0 < 0, ({})'.format(r0)
    assert z0 <= z1, 'z0 > z1, ({} > {})'.format(z0, z1)

    if cn is not None:
        volume = (z1 - z0) * np.pi * (r1**2 - r0**2)
        no = int(volume * cn)

    if no < 1:
        return np.zeros((3, 0))

    no = int(no)

    phi = np.random.random(no)
    r = np.random.random(no)
    z = np.random.random(no)

    phi = phi * 2 * np.pi
    r = np.sqrt(r * (r1**2 - r0**2) + r0**2)

    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = z * (z1 - z0) + z0

    return np.vstack((x, y, z))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       SORT                                                          #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def find_nearest(pos_i, pos_j=None, no=None):
    ''' For each element in pos_i;
    find the other point in pos_i (pos_j) which is closest.

    Parameters
    ----------
    pos_i : numpy array (3, -1)
            origin positions
    pos_j : numpy array (3, -1)
            candidate positions
    no    : int
            0 for nearest, 1 for second nearest, and so on

    Returns
    -------
    nn_idx : int
             index of n'th nearest
    nn_dst : float
             distance to n'th nearest
    '''

    # Set default for 'no'.
    if no is None:
        if pos_j is None:
            # Choose second closest element when only one list is given.
            no = 1
        else:
            # Choose closest element if a candidate list is given.
            no = 0

    (M_idx, M_dst) = sort_nearest(pos_i, pos_j=pos_j)

    nn_idx = M_idx[:, no]
    # nn_dst = M_dst[:, no]  # Warning! Wrong method! The below is correct!
    nn_dst = M_dst[:, 0] * 0  # Weird solution, but worked for edge cases.
    for i in range(pos_i.shape[1]):
        nn_dst[i] = M_dst[i, nn_idx[i]]

    return (nn_idx, nn_dst)


def sort_nearest(pos_i, pos_j=None):
    ''' For each element in pos_i;
    find the other point in pos_i (pos_j) which is closest.

    Parameters
    ----------
    pos_i : numpy array (3, -1)
            origin positions
    pos_j : numpy array (3, -1)
            candidate positions

    Returns
    -------
    M_idx : mat[int]
            index of the j'th closest to i
    M_dst : mat[float]
            abs distance from i to j
    '''

    # It is also possible to make a bigger matrix:
    # M = in1.reshape(3, 1, -1) - in2.reshape(3, -1, 1)
    # D = np.linalg.norm(M, axis=0)
    # D is now a matrix over distances from in1 to in2
    # Finding the lowest number and position on each row or column
    # gives the output.
    # Squaring, and only taking the root of the lowest number might be faster.
    # Ensure parameters of correct size.

    if pos_j is None:
        pos_j = pos_i

    assert pos_i.shape[0] == 3, 'pos_i.shape[0] =! 3'
    assert pos_j.shape[0] == 3, 'pos_i.shape[0] =! 3'

    li = pos_i.shape[1]
    lj = pos_j.shape[1]

    # Initiate output
    M_idx = np.zeros((li, lj)).astype(int)
    M_dst = np.zeros((li, lj))

    # Populate output
    for i in range(li):
        r_pos = pos_j - pos_i[:, i].reshape(3, -1)

        M_dst[i, :] = np.linalg.norm(r_pos, axis=0)
        M_idx[i, :] = np.argsort(M_dst[i, :])

    return (M_idx, M_dst)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       GET INFO                                                      #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def get_min_max(pos):
    ''' Find certain properties of an array of positions.

    Parameters
    ----------
    pos :   numpy array (3, -1)
            positions

    Returns
    -------
    out :   dict
            min/max for r, x, y, z
    '''

    # pos.reshape(3,-1)
    if (pos.shape[0] != 3) or (pos.shape[1] == 0):
        return {}

    x = pos[0, :]
    y = pos[1, :]
    z = pos[2, :]
    r2 = (x**2 + y**2 + z**2)

    out = {}
    out['r0'] = np.sqrt(r2.min())
    out['r1'] = np.sqrt(r2.max())
    out['x0'] = x.min()
    out['x1'] = x.max()
    out['y0'] = y.min()
    out['y1'] = y.max()
    out['z0'] = z.min()
    out['z1'] = z.max()

    return out


#
