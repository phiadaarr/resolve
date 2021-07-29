# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import numpy as np

import nifty8 as ift
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

    oplist = [ift.GaussianEnergy(domain=dom["a"]) @ Extract(dom, "a") @ ift.ScalingOperator(dom, 2.).exp() for ii in range(hi-lo)]
    op = rve.SliceSum(oplist, lo, parallel_space, rve.mpi.comm)
    ift.extra.check_operator(op, ift.from_random(op.domain))
