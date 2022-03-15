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
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np

import resolve as rve


class Extract(ift.LinearOperator):
    def __init__(self, domain, key):
        self._domain = ift.MultiDomain.make(domain)
        self._target = self._domain[key]
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._key = str(key)

    def apply(self,x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return x[self._key]
        else:
            return ift.makeField(ift.makeDomain({self._key: self._domain[self._key]}), {self._key: x.val}).unite(ift.full(self.domain, 0.))


def test_slice_sum():
    parallel_space = ift.UnstructuredDomain(10)
    dom = {"a": ift.RGSpace(20), "b": ift.RGSpace(12)}
    lo, hi = ift.utilities.shareRange(parallel_space.size,
                                      rve.mpi.ntask,
                                      rve.mpi.rank)
    oplist = [Extract(dom, "a") @ ift.ScalingOperator(dom, 2.).exp() for ii in range(hi-lo)]
    op = rve.SliceSum(oplist, lo, parallel_space, rve.mpi.comm)
    ift.extra.check_operator(op, ift.from_random(op.domain))

    oplist = [ift.GaussianEnergy(domain=dom["a"], sampling_dtype=np.float64) @ Extract(dom, "a") @ ift.ScalingOperator(dom, 2.).exp() for ii in range(hi-lo)]
    op = rve.SliceSum(oplist, lo, parallel_space, rve.mpi.comm)
    ift.extra.check_operator(op, ift.from_random(op.domain))
    op(ift.Linearization.make_var(ift.from_random(op.domain), True)).metric.draw_sample()
