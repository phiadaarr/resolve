# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

from .util import my_assert

TABLE = {
    5: "RR",
    6: "RL",
    7: "LR",
    8: "LL",
    9: "XX",
    10: "XY",
    11: "YX",
    12: "YY"
}
INVTABLE = {val: key for key, val in TABLE.items()}


class Polarization:
    def __init__(self, indices):
        self._ind = list(indices)
        my_assert(len(self._ind) <= 4)

    def circular(self):
        if set(self._ind) <= set([5, 6, 7, 8]):
            return True
        if set(self._ind) <= set([9, 10, 11, 12]):
            return False
        raise RuntimeError

    def stokes_i_indices(self):
        keys = ["LL", "RR"] if self.circular else ["XX", "YY"]
        return [self._ind.index(self.INVTABLE[kk]) for kk in keys]

    def __len__(self):
        return len(self._ind)
