# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

from .util import compare_attributes, my_asserteq


class Direction:
    """
    Parameters
    ----------
    phase_center : list of float
        coordinate of phase center
    equinox : int
        reference year of the equinox
    """

    def __init__(self, phase_center, equinox):
        my_asserteq(len(phase_center), 2)
        self._pc = phase_center
        self._e = float(equinox)

    @property
    def phase_center(self):
        return self._pc

    @property
    def equinox(self):
        return self._e

    def __repr__(self):
        return f"Direction({self._pc}, equinox {self._e})"

    def to_list(self):
        return [*self._pc, self._e]

    @staticmethod
    def from_list(lst):
        return Direction(lst[0:2], lst[2])

    def __eq__(self, other):
        if not isinstance(other, Direction):
            return False
        return compare_attributes(self, other, ("_pc", "_e"))


class Directions:
    def __init__(self, phase_centers, equinox):
        assert phase_centers.ndim == 2
        assert phase_centers.shape[1] == 2
        self._pc = phase_centers
        self._e = float(equinox)

    @property
    def phase_centers(self):
        return self._pc

    @property
    def equinox(self):
        return self._e

    def __repr__(self):
        return f"Directions({self._pc}, equinox={self._e})"

    def to_list(self):
        return [self._pc, self._e]

    def __len__(self):
        return self._pc.shape[0]

    @staticmethod
    def from_list(lst):
        return Directions(lst[0], lst[1])

    def __eq__(self, other):
        if not isinstance(other, Direction):
            return False
        return compare_attributes(self, other, ("_pc", "_e"))

    def __getitem__(self, slc):
        return Directions(self._pc[slc], self._e)
