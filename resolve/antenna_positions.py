# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

from .util import compare_attributes, my_assert, my_asserteq


class AntennaPositions:
    """Summarizes all information on antennas and baselines. If calibration is
    performed this class stores also antenna indices and time information.
    For imaging only this is not necessary."""

    # FIXME Split this class into two. One for only imaging, one also for calibration
    def __init__(self, uvw, ant1=None, ant2=None, time=None):
        if ant1 is None:
            my_asserteq(ant2, time, None)
            my_asserteq(uvw.ndim, 2)
            my_asserteq(uvw.shape[1], 3)
        else:
            my_asserteq(ant1.shape, ant2.shape, time.shape)
            my_asserteq(uvw.shape, (ant1.size, 3))
            my_assert(np.issubdtype(ant1.dtype, np.integer))
            my_assert(np.issubdtype(ant2.dtype, np.integer))
            my_assert(np.issubdtype(time.dtype, np.floating))
        my_assert(np.issubdtype(uvw.dtype, np.floating))
        self._uvw, self._time = uvw, time
        self._ant1, self._ant2 = ant1, ant2
        self._t0 = None

    @property
    def only_imaging(self):
        return self._ant1 is None

    def to_list(self):
        return [self._uvw, self._ant1, self._ant2, self._time]

    def unique_antennas(self):
        if self.only_imaging:
            raise RuntimeError
        return set(np.unique(self._ant1)) | set(np.unique(self._ant2))

    def unique_times(self):
        if self.only_imaging:
            raise RuntimeError
        return set(np.unique(self._time))

    @staticmethod
    def from_list(lst):
        return AntennaPositions(*lst)

    def move_time(self, t0):
        if self.only_imaging:
            raise RuntimeError
        return AntennaPositions(self._uvw, self._ant1, self._ant2, self._time + t0)

    def __eq__(self, other):
        if not isinstance(other, AntennaPositions):
            return False
        return compare_attributes(self, other, ("_uvw", "_time", "_ant1", "_ant2"))

    def __len__(self):
        return self._uvw.shape[0]

    def __getitem__(self, slc):
        if self.only_imaging:
            return AntennaPositions(self._uvw[slc])
        return AntennaPositions(
            self._uvw[slc], self._ant1[slc], self._ant2[slc], self._time[slc]
        )

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
