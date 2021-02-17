# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift


class AllreduceSum(ift.Operator):
    def __init__(self, oplist, comm, nwork=None):
        """nwork only needed if samples need to be drawn and oplist are EnergyOperators."""
        self._oplist, self._comm = oplist, comm
        self._domain = self._oplist[0].domain
        self._target = self._oplist[0].target
        self._nwork = nwork

    def apply(self, x):
        self._check_input(x)
        if not ift.is_linearization(x):
            return ift.utilities.allreduce_sum(
                [op(x) for op in self._oplist], self._comm
            )
        opx = [op(x) for op in self._oplist]
        val = ift.utilities.allreduce_sum([lin.val for lin in opx], self._comm)
        jac = AllreduceSumLinear([lin.jac for lin in opx], self._comm)
        if opx[0].metric is None:
            return x.new(val, jac)
        met = AllreduceSumLinear([lin.metric for lin in opx], self._comm, self._nwork)
        return x.new(val, jac, met)


class AllreduceSumLinear(ift.LinearOperator):
    def __init__(self, oplist, comm=None, nwork=None):
        assert all(isinstance(oo, ift.LinearOperator) for oo in oplist)
        self._domain = ift.makeDomain(oplist[0].domain)
        self._target = ift.makeDomain(oplist[0].target)
        cap = oplist[0]._capability
        assert all(oo.domain == self._domain for oo in oplist)
        assert all(oo.target == self._target for oo in oplist)
        assert all(oo._capability == cap for oo in oplist)
        self._capability = (self.TIMES | self.ADJOINT_TIMES) & cap
        self._oplist = oplist
        self._comm = comm
        self._nwork = nwork

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.utilities.allreduce_sum(
            [op.apply(x, mode) for op in self._oplist], self._comm
        )

    def draw_sample(self, from_inverse=False):
        size, rank, _ = ift.utilities.get_MPI_params_from_comm(self._comm)
        lo, _ = ift.utilities.shareRange(self._nwork, size, rank)
        sseq = ift.random.spawn_sseq(self._nwork)
        local_samples = []
        for ii, op in enumerate(self._oplist):
            with ift.random.Context(sseq[lo + ii]):
                local_samples.append(op.draw_sample(from_inverse))
        return ift.utilities.allreduce_sum(local_samples, self._comm)
