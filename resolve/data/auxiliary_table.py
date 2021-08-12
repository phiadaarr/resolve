# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

from ..util import compare_attributes, my_assert, my_assert_isinstance, my_asserteq


class AuxiliaryTable:
    def __init__(self, inp_dict):
        my_assert_isinstance(inp_dict, dict)
        nrow = None
        for kk, lst in inp_dict.items():
            my_assert_isinstance(kk, str)
            my_assert_isinstance(lst, (list, np.ndarray))
            if nrow is None:
                nrow = len(lst)
            my_asserteq(nrow, len(lst))
            my_asserteq(type(elem) for elem in lst)
        self._dct = inp_dict
        self._eq_attributes = [self._dct]

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        return compare_attributes(self, other, self._eq_attributes)

    def __str__(self):
        s = ["AuxiliaryTable:"]
        for kk, lst in self._dct.items():
            if isinstance(lst, np.ndarray):
                s.append(f"  {kk:<20} {str(lst.shape):>10} {str(lst.dtype):>15}")
            else:
                s.append(f"  {kk:<20} {len(lst):>10} {str(type(lst[0])):>15}")
        return "\n".join(s)

    def to_list(self):
        return [list(self._dct.keys())] + [ss for ss in self._dct.values()]

    @staticmethod
    def from_list(lst):
        keys = lst[0]
        dct = {kk: lst[ii+1] for ii, kk in enumerate(keys)}
        return AuxiliaryTable(dct)
