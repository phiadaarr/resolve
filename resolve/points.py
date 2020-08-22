# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift
from .util import my_asserteq, my_assert_isinstance


class PointInserter(ift.LinearOperator):
    def __init__(self, target, positions):
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        positions = np.array(positions)
        my_asserteq(len(self._target.shape), 2)
        my_asserteq(len(self._target), 1)
        my_assert_isinstance(self._target[0], ift.RGSpace)
        my_asserteq(len(positions.shape), 2)
        my_asserteq(positions.shape[1], 2)
        dx = np.array(self._target[0].distances)
        center = np.array(self._target[0].shape)//2
        self._inds = np.unique(np.round(positions/dx + center).astype(int).T, axis=1)
        self._domain = ift.makeDomain(ift.UnstructuredDomain(self._inds.shape[1]))

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        xs, ys = self._inds
        if mode == self.TIMES:
            res = np.zeros(self._target.shape, dtype=x.dtype)
            res[xs, ys] = x
        else:
            res = x[xs, ys]
        return ift.makeField(self._tgt(mode), res)
