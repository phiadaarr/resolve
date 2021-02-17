# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift


class AllreduceSum(ift.Operator):
    def __init__(self, oplist, comm):
        self._oplist, self._comm = oplist, comm
        self._domain = ift.makeDomain(
            _get_global_unique(oplist, lambda op: op.domain, comm)
        )
        self._target = ift.makeDomain(
            _get_global_unique(oplist, lambda op: op.target, comm)
        )

    def apply(self, x):
        self._check_input(x)
        if not ift.is_linearization(x):
            return ift.utilities.allreduce_sum(
                [op(x) for op in self._oplist], self._comm
            )
        opx = [op(x) for op in self._oplist]
        val = ift.utilities.allreduce_sum([lin.val for lin in opx], self._comm)
        jac = AllreduceSumLinear([lin.jac for lin in opx], self._comm)
        if _get_global_unique(opx, lambda op: op.metric is None, self._comm):
            return x.new(val, jac)
        met = AllreduceSumLinear([lin.metric for lin in opx], self._comm)
        return x.new(val, jac, met)


class AllreduceSumLinear(ift.LinearOperator):
    def __init__(self, oplist, comm=None):
        assert all(isinstance(oo, ift.LinearOperator) for oo in oplist)
        self._domain = ift.makeDomain(
            _get_global_unique(oplist, lambda op: op.domain, comm)
        )
        self._target = ift.makeDomain(
            _get_global_unique(oplist, lambda op: op.target, comm)
        )
        cap = _get_global_unique(oplist, lambda op: op._capability, comm)
        self._capability = (self.TIMES | self.ADJOINT_TIMES) & cap
        self._oplist = oplist
        self._comm = comm
        local_nwork = [len(oplist)] if comm is None else comm.allgather(len(oplist))
        size, rank, _ = ift.utilities.get_MPI_params_from_comm(comm)
        self._nwork = sum(local_nwork)
        self._lo = ([0] + list(np.cumsum(local_nwork)))[rank]

    def apply(self, x, mode):
        self._check_input(x, mode)
        lst = [op.apply(x, mode) for op in self._oplist]
        return ift.utilities.allreduce_sum(lst, self._comm)

    def draw_sample(self, from_inverse=False):
        sseq = ift.random.spawn_sseq(self._nwork)
        local_samples = []
        for ii, op in enumerate(self._oplist):
            with ift.random.Context(sseq[self._lo + ii]):
                local_samples.append(op.draw_sample(from_inverse))
        return ift.utilities.allreduce_sum(local_samples, self._comm)


def _get_global_unique(lst, f, comm):
    caps = [f(oo) for oo in lst]
    if comm is not None:
        caps = comm.allgather(caps)
        caps = [aa for cc in caps for aa in cc]
    cap = caps[0]
    assert all(cc == cap for cc in caps)
    return cap
