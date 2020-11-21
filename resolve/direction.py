# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

from .util import compare_attributes, my_asserteq


class Direction:
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
