# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

from .util import (compare_attributes, my_assert, my_assert_isinstance,
                   my_asserteq)


class AntennaPositions:
    def __init__(self, uvw, ant1, ant2, time):
        my_asserteq(ant1.shape, ant2.shape, time.shape)
        my_asserteq(uvw.shape, (ant1.size, 3))
        my_assert(np.issubdtype(uvw.dtype, np.floating))
        my_assert(np.issubdtype(ant1.dtype, np.integer))
        my_assert(np.issubdtype(ant2.dtype, np.integer))
        my_assert(np.issubdtype(time.dtype, np.floating))
        self._uvw, self._time = uvw, time
        self._ant1, self._ant2 = ant1, ant2
        self._t0 = None

    def to_list(self):
        return [self._uvw, self._ant1, self._ant2, self._time]

    def unique_antennas(self):
        return set(np.unique(self._ant1)) | set(np.unique(self._ant2))

    def unique_times(self):
        return set(np.unique(self._time))

    @staticmethod
    def from_list(lst):
        return AntennaPositions(*lst)

    def move_time(self, t0):
        return AntennaPositions(self._uvw, self._ant1, self._ant2,
                                self._time+t0)

    def __eq__(self, other):
        if not isinstance(other, AntennaPositions):
            return False
        return compare_attributes(self, other, ('_uvw', '_time', '_ant1', '_ant2'))

    def __len__(self):
        return self._ant1.size

    def __getitem__(self, slc):
        return AntennaPositions(self._uvw[slc], self._ant1[slc],
                                self._ant2[slc], self._time[slc])

    @property
    def uvw(self):
        return self._uvw

    @property
    def time(self):
        return self._time

    @property
    def ant1(self):
        return self._ant1

    @property
    def ant2(self):
        return self._ant2
