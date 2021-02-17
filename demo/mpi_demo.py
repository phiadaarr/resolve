# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society

from functools import reduce
from operator import add

import numpy as np

import nifty7 as ift


def getop(comm):
    """Return energy operator that maps the full multi-frequency sky onto
    the log-likelihood value for a frequency slice."""

    def get_local_oplist(comm):
        d = np.load("data.npy")
        invcov = np.load("invcov.npy")
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
        return lst

    if comm == -1:
        d = np.load("data.npy")
        invcov = np.load("invcov.npy")
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
        op = AllreduceSum(get_local_oplist, comm)

    ift.extra.check_operator(op, ift.from_random(op.domain))

    sky = ift.FieldAdapter(op.domain, "sky")
    return op @ sky.exp()


class AllreduceSum(ift.Operator):
    def __init__(self, get_oplist, comm):
        self._comm = comm
        self._oplist = get_oplist(comm)
        self._domain = self._oplist[0].domain
        self._target = self._oplist[0].target

    def apply(self, x):
        self._check_input(x)
        return ift.utilities.allreduce_sum([op(x) for op in self._oplist], self._comm)


def allclose(gen):
    from types import GeneratorType

    ref = next(gen) if isinstance(gen, GeneratorType) else gen[0]
    for aa in gen:
        print(aa)
        ift.extra.assert_allclose(ref, aa)


def main():
    ddomain = ift.UnstructuredDomain(7), ift.UnstructuredDomain(5)
    comm, size, rank, master = ift.utilities.get_MPI_params()
    data = ift.from_random(ddomain)
    invcov = ift.from_random(ddomain).exp()
    if master:
        np.save("data.npy", data.val)
        np.save("invcov.npy", invcov.val)
    if comm is not None:
        comm.Barrier()

    lh0 = getop(-1)
    lh1 = getop(None)
    lh2 = getop(comm)
    lhs = lh0, lh1, lh2
    hams = tuple(
        ift.StandardHamiltonian(lh, ift.GradientNormController(iteration_limit=10))
        for lh in lhs
    )

    # Evaluate Field
    pos = ift.from_random(lh0.domain)
    allclose(op(pos) for op in lhs)

    # Evaluate Linearization
    for wm in [False, True]:
        lin = ift.Linearization.make_var(pos, wm)
        allclose(lh(lin).val for lh in lhs)
        allclose(lh(lin).gradient for lh in lhs)

        for _ in range(10):
            foo = ift.Linearization.make_var(ift.from_random(lh0.domain), wm)
            bar = ift.from_random(lh0.domain)
            allclose(lh(foo).jac(bar) for lh in lhs)
            if wm:
                allclose(lh(foo).metric(bar) for lh in lhs)
            bar = ift.from_random(lh0.target)
            allclose(lh(foo).jac.adjoint(bar) for lh in lhs)

        # Minimization
        pos = ift.from_random(lh0.domain)
        es = tuple(ift.EnergyAdapter(pos, ham, want_metric=wm) for ham in hams)
        ic = ift.GradientNormController(iteration_limit=5)
        mini = ift.NewtonCG(ic) if wm else ift.SteepestDescent(ic)
        allclose(mini(e)[0].position for e in es)

        # Draw samples
        lin = ift.Linearization.make_var(ift.from_random(lh0.domain), True)
        samps_lh = []
        samps_ham = []
        for ii in range(len(lhs)):
            with ift.random.Context(42):
                samps_lh.append(lhs[ii](lin).metric.draw_sample())
                samps_ham.append(hams[ii](lin).metric.draw_sample())
        allclose(samps_lh)
        allclose(samps_ham)

        allclose(
            mini(ift.MetricGaussianKL.make(pos, ham, 3, True))[0].position for ham in hams
        )

if __name__ == "__main__":
    main()
