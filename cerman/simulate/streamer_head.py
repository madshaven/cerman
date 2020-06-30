#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Streamer Head.

Derived from ElectricalHyperboloid.
Consider a change of implementation to allow other types.
'''

# General imports
import numpy as np
import logging

# Import from project files
from ..core.eh_list import EHList
from ..core.electrical_hyperboloid import ElectricalHyperboloid

# settings
ooi = np.array([0, 0, 1]).reshape(3, -1)
eps = np.finfo(float).eps  # 2.22e-16 for double

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       STREAMER HEAD                                                 #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class StreamerHead(ElectricalHyperboloid):

    def __init__(self, pos, rp, U0=0, k=1, dtype=None):
        super().__init__(pos=pos, rp=rp, U0=U0, k=k, dtype=dtype)

    def __repr__(self):
        msg = 'StreamerHead(pos={}, rp={}, U0={}, k={}, dtype={})'
        msg = msg.format(self.pos, self.rp, self.U0, self.k, self.dtype)
        return msg


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                                                     #
#       STREAMER HEAD LIST                                            #
#                                                                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


class SHList(EHList):

    _item_class = StreamerHead


#
