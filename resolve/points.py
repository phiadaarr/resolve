# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np

from .util import assert_sky_domain


class PointInserter(ift.LinearOperator):
    def __init__(self, target, positions):
        self._target = ift.DomainTuple.make(target)
        assert_sky_domain(self._target)
        pdom, tdom, fdom, sdom = self._target
        self._capability = self.TIMES | self.ADJOINT_TIMES
        positions = np.array(positions)
        dx = np.array(sdom.distances)
        center = np.array(sdom.shape) // 2
        self._inds = np.unique(np.round(positions / dx + center).astype(int).T, axis=1)
        npoints = self._inds.shape[1]
        if npoints != len(positions):
            print("WARNING: Resolution not sufficient to assign a unique pixel to every point source.")
        self._domain = ift.makeDomain((pdom, tdom, fdom, ift.UnstructuredDomain(npoints)))

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        xs, ys = self._inds
        if mode == self.TIMES:
            res = np.zeros(self._target.shape, dtype=x.dtype)
            res[..., xs, ys] = x
        else:
            res = np.sum(x, axis=(1, 2))
            res = res[:, xs, ys]
            res = res[:, None, None]
        return ift.makeField(self._tgt(mode), res)
