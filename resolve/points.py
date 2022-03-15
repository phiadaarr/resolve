# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
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
            res = x[..., xs, ys]
        return ift.makeField(self._tgt(mode), res)
