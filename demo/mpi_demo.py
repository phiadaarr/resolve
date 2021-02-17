# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society

from functools import reduce
from operator import add
from types import GeneratorType

import numpy as np

import nifty7 as ift


def getop(comm):
    """Return energy operator that maps the full multi-frequency sky onto
    the log-likelihood value for a frequency slice."""

    d = np.load("data.npy")
    invcov = np.load("invcov.npy")
    if comm == -1:
        nwork = d.shape[0]
        ddom = ift.UnstructuredDomain(d[0].shape)
        ops = [
            ift.GaussianEnergy(
                ift.makeField(ddom, d[ii]), ift.makeOp(ift.makeField(ddom, invcov[ii]))
            )
            for ii in range(nwork)
        ]
        op = reduce(add, ops)
    else:
        nwork = d.shape[0]
        size, rank, _ = ift.utilities.get_MPI_params_from_comm(comm)
        lo, hi = ift.utilities.shareRange(nwork, size, rank)
        local_indices = range(lo, hi)
        lst = []
        for ii in local_indices:
            ddom = ift.UnstructuredDomain(d[ii].shape)
            dd = ift.makeField(ddom, d[ii])
            iicc = ift.makeOp(ift.makeField(ddom, invcov[ii]))
            ee = ift.GaussianEnergy(dd, iicc)
            lst.append(ee)
        op = AllreduceSum(lst, comm, nwork)
    ift.extra.check_operator(op, ift.from_random(op.domain))
    sky = ift.FieldAdapter(op.domain, "sky")
    return op @ sky.exp()


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


def allclose(gen):
    ref = next(gen) if isinstance(gen, GeneratorType) else gen[0]
    for aa in gen:
        ift.extra.assert_allclose(ref, aa)


def main():
    ddomain = ift.UnstructuredDomain(4), ift.UnstructuredDomain(1)
    comm, size, rank, master = ift.utilities.get_MPI_params()
    data = ift.from_random(ddomain)
    invcov = ift.from_random(ddomain).exp()
    if master:
        np.save("data.npy", data.val)
        np.save("invcov.npy", invcov.val)
    if comm is not None:
        comm.Barrier()

    lhs = getop(-1), getop(None), getop(comm)
    hams = tuple(
        ift.StandardHamiltonian(lh, ift.GradientNormController(iteration_limit=10))
        for lh in lhs
    )
    lhs_for_sampling = lhs[1:]
    hams_for_sampling = hams[1:]

    # Evaluate Field
    dom, tgt = lhs[0].domain, lhs[0].target
    pos = ift.from_random(dom)
    allclose(op(pos) for op in lhs)

    # Evaluate Linearization
    for wm in [False, True]:
        lin = ift.Linearization.make_var(pos, wm)
        allclose(lh(lin).val for lh in lhs)
        allclose(lh(lin).gradient for lh in lhs)

        for _ in range(10):
            foo = ift.Linearization.make_var(ift.from_random(dom), wm)
            bar = ift.from_random(dom)
            allclose(lh(foo).jac(bar) for lh in lhs)
            if wm:
                allclose(lh(foo).metric(bar) for lh in lhs)
            bar = ift.from_random(tgt)
            allclose(lh(foo).jac.adjoint(bar) for lh in lhs)

        # Minimization
        pos = ift.from_random(dom)
        es = tuple(ift.EnergyAdapter(pos, ham, want_metric=wm) for ham in hams)
        ic = ift.GradientNormController(iteration_limit=5)
        mini = ift.NewtonCG(ic) if wm else ift.SteepestDescent(ic)
        allclose(mini(e)[0].position for e in es)

        # Draw samples
        lin = ift.Linearization.make_var(ift.from_random(dom), True)
        samps_lh, samps_ham = [], []
        for ii, (llhh, hh) in enumerate(zip(lhs_for_sampling, hams_for_sampling)):
            with ift.random.Context(42):
                samps_lh.append(llhh(lin).metric.draw_sample())
            with ift.random.Context(42):
                samps_ham.append(hh(lin).metric.draw_sample())
        allclose(samps_lh)
        allclose(samps_ham)

        mini_results = []
        for ham in hams_for_sampling:
            with ift.random.Context(42):
                mini_results.append(
                    mini(ift.MetricGaussianKL.make(pos, ham, 3, True))[0].position
                )
        allclose(mini_results)


if __name__ == "__main__":
    main()
