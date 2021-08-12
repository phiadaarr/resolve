# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift

from ..util import compare_attributes, my_assert

TABLE = {5: "RR", 6: "RL", 7: "LR", 8: "LL", 9: "XX", 10: "XY", 11: "YX", 12: "YY"}
INVTABLE = {val: key for key, val in TABLE.items()}


class Polarization:
    """Stores polarization information of a data set and is used for
    translating between indices (5, 6, 7, 8, ...) and strings
    ("RR", "RL", "LR", "LL", ...).

    Parameters
    ----------
    indices : tuple of polarization indices.
        Takes integer values between 5 and including 12.
    """

    def __init__(self, indices):
        self._ind = tuple(indices)
        my_assert(len(self._ind) <= 4)
        self._trivial = len(indices) == 0

    @staticmethod
    def trivial():
        return Polarization([])

    def restrict_to_stokes_i(self):
        inds = (8, 5) if self.circular() else (9, 12)
        return Polarization(inds)

    def restrict_by_name(self, lst):
        return Polarization([INVTABLE[ss] for ss in lst])

    def circular(self):
        if self._trivial:
            raise RuntimeError
        if set(self._ind) <= set([5, 6, 7, 8]):
            return True
        if set(self._ind) <= set([9, 10, 11, 12]):
            return False
        raise RuntimeError

    def has_crosshanded(self):
        if len(set(self._ind) & set([6, 7, 10, 11])) > 0:
            return True
        return False

    def stokes_i_indices(self):
        if self._trivial:
            raise RuntimeError
        keys = ["LL", "RR"] if self.circular() else ["XX", "YY"]
        return [self._ind.index(INVTABLE[kk]) for kk in keys]

    def __len__(self):
        if self._trivial:
            return 1
        return len(self._ind)

    def to_list(self):
        return list(self._ind)

    def to_str_list(self):
        return [TABLE[ii] for ii in self._ind]

    @staticmethod
    def from_list(lst):
        return Polarization(lst)

    def __eq__(self, other):
        if not isinstance(other, Polarization):
            return False
        return compare_attributes(self, other, ("_ind",))

    def __repr__(self):
        return f"Polarization({self._ind})"