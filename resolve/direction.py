# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

from .util import my_assert


class Direction:
    def __init__(self, phase_center, equinox):
        my_assert(len(phase_center) == 2)
        equinox = str(equinox)[1:]
        if equinox == "1950_VLA":
            equinox = 1950
        self._e = float(equinox)
        self._pc = phase_center
        self._e = equinox

    @property
    def phase_center(self):
        return self._pc

    @property
    def equinox(self):
        return self._e

    def __repr__(self):
        return f'Direction({self._pc}, equinox {self._e})'
