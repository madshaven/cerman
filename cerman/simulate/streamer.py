#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' This module contains a class for keeping a list of StreamerHead objects.
    Its main purpose is to keep track of heads added and removed.
'''

# General imports
import numpy as np
import logging

# import from project files
from .streamer_head import StreamerHead
from .streamer_head import SHList

# settings
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# idea: add tau as property
# idea: add capacitance as property


class Streamer():
    def __init__(self):
        # note:
        # "_heads*" are lists of streamer heads
        # corresponding properties without "_" are SHLists

        # Maintained variables
        self._heads = []            # List of all heads
        self.head_z_min = None      # Tip closest to plane
        self.head_r_max = None      # Tip furthest away from center
        self.z_min = 1e10           # Distance for head closest to plane
        self.r_max = 0              # Tip maximum distance from z-axis

        # Variables cleaned/reset every iteration
        self._heads_appended_all = []     # Tips added last iteration
        self._heads_removed_all = []      # Tips removed last iteration

        self._add_properties()      # Add properties

        logger.debug('Initiated Streamer')
        # idea: implement a __str__
        logger.log(5, 'Streamer.__dict__')
        for k, v in self.__dict__.items():
            logger.log(5, '  "{}": {}'.format(k, v))

    def clean(self):
        '''Clean at end of iteration. '''
        diff = self.no_appended_all - self.no_removed_all
        if abs(diff) > 0:
            msg = 'Changed number of streamer heads {}'
            logger.debug(msg.format(diff))
        self._heads_appended_all    = []
        self._heads_removed_all     = []

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       DERIVED ATTIBUTES                                         #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # properties derived from SHList
    U0 = property(lambda self: self.heads.U0)
    U = property(lambda self: self.heads.U)
    k = property(lambda self: self.heads.k)

    # ensure that these properties are always correct lists
    heads = property(lambda self: SHList(self._heads))
    heads_appended_all = property(
        lambda self: SHList(self._heads_appended_all))
    heads_removed_all = property(
        lambda self: SHList(self._heads_removed_all))

    # new heads that are kept
    heads_appended = property(lambda self: SHList([
        h for h in self._heads_appended_all if h in self._heads]))

    # old heads that are removed
    heads_removed = property(lambda self: SHList([
        h for h in self._heads_removed_all
        if h not in self._heads_appended_all]))

    # positions
    def tl2p(self, tl):   # h list to array of positions
        pl = [h.pos for h in tl]  # list positions
        return np.hstack([np.zeros((3, 0))] + pl)  # safe for empty lists

    _properties_added = False

    @classmethod
    def _add_properties(cls):
        # dynamically add properties after the class is created
        # not possible during class creation
        if cls._properties_added:
            return
        else:
            cls._properties_added = True

        # each of these keys corresponds to a SHList
        # a property for no, pos, and dict
        # is created for each list
        _properties_keys = [
            'heads',
            'heads_appended',
            'heads_removed',
            'heads_appended_all',
            'heads_removed_all',
            ]

        def _add(key):
            # add attribute `no` for each property
            setattr(cls, key.replace('heads', 'no'), property(
                lambda self: len(getattr(self, key))))
            # add attribute `pos` for each property
            setattr(cls, key.replace('heads', 'pos'), property(
                lambda self: self.tl2p(getattr(self, key))))
            # add attribute `heads_dict` for each property
            setattr(cls, key.replace('heads', 'heads_dict'), property(
                lambda self: [h.to_dict() for h in getattr(self, key)]))

        for _key in _properties_keys:
            _add(_key)  # pass to function ensure new "key" variable
                        # for each function call

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       APPEND/REMOVE                                             #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _append_one(self, sh):
        '''Append one streamer head.'''
        if type(sh) is not StreamerHead:
            logger.error('Cannot append non-StreamerHead element.')
            return
        self._heads.append(sh)
        self._heads_appended_all.append(sh)
        if sh.d < self.z_min:
            self.z_min = sh.d
            self.head_z_min = sh
        if sh.r > self.r_max:
            self.r_max = sh.r
            self.head_r_max = sh

    def append(self, heads):
        '''Append a list of heads.'''
        if type(heads) is StreamerHead:
            heads = [heads]
        for head in heads:
            self._append_one(head)

    def _remove_one(self, head):
        '''Remove one head.'''
        if head not in self._heads:
            msg = 'Tried to remove head not in list {}, type: {}'
            logger.warning(msg.format(head, type(head)))
            return

        self._heads.remove(head)
        self._heads_removed_all.append(head)

        if head.d <= self.z_min:
            msg = 'Removed leading head'
            logger.debug(msg)

        if head == self.head_r_max:
            self.r_max = 0
            for head in self._heads:
                if head.r > self.r_max:
                    self.r_max = head.r
                    self.head_r_max = head

    def remove(self, heads):
        '''Remove a list of heads.'''
        if type(heads) is StreamerHead:
            heads = [heads]
        for head in heads:
            self._remove_one(head)


#
