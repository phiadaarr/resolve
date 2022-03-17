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
from types import GeneratorType

import nifty8 as ift
import numpy as np
from mpi4py import MPI

import resolve as rve


def getop(comm, typ):
    """Return energy operator that maps the full multi-frequency sky onto
    the log-likelihood value for a frequency slice."""

    d = np.load("data.npy")
    invcov = np.load("invcov.npy")
    skydom = ift.UnstructuredDomain(d.shape[0]), ift.UnstructuredDomain(d.shape[1:])
    if comm == -1:
        nwork = d.shape[0]
        ddom = ift.UnstructuredDomain(d[0].shape)
        ops = [
            ift.GaussianEnergy(
                ift.makeField(ddom, d[ii]), ift.makeOp(ift.makeField(ddom, invcov[ii]), sampling_dtype=d[ii].dtype)
            )
            @ ift.DomainTupleFieldInserter(skydom, 0, (ii,)).adjoint
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
            iicc = ift.makeOp(ift.makeField(ddom, invcov[ii]), sampling_dtype=d[ii].dtype)
            ee = ift.GaussianEnergy(dd, iicc)
            if typ == 0:
                ee = ee @ ift.DomainTupleFieldInserter(skydom, 0, (ii,)).adjoint
            lst.append(ee)
        if typ == 0:
            op = rve.AllreduceSum(lst, comm)
        else:
            op = rve.SliceSum(lst, lo, skydom[0], comm)
    ift.extra.check_operator(op, ift.from_random(op.domain))
    sky = ift.FieldAdapter(skydom, "sky")
    return op @ sky.exp()


def allclose(gen):
    ref = next(gen) if isinstance(gen, GeneratorType) else gen[0]
    for aa in gen:
        ift.extra.assert_allclose(ref, aa)


def test_mpi_adder():
    # FIXME Write tests for non-EnergyOperators and linear operators.
    ddomain = ift.UnstructuredDomain(4), ift.UnstructuredDomain(1)
    comm, size, rank, master = ift.utilities.get_MPI_params()
    data = ift.from_random(ddomain)
    invcov = ift.from_random(ddomain).exp()
    if master:
        np.save("data.npy", data.val)
        np.save("invcov.npy", invcov.val)
    if comm is not None:
        comm.Barrier()

    lhs = (
        getop(-1, 0),
        getop(-1, 1),
        getop(None, 0),
        getop(None, 1),
        getop(comm, 0),
        getop(comm, 1),
    )
    hams = tuple(
        ift.StandardHamiltonian(lh, ift.GradientNormController(iteration_limit=10))
        for lh in lhs
    )
    lhs_for_sampling = lhs[2:]
    hams_for_sampling = hams[2:]

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
            print(ii)
            with ift.random.Context(42):
                samps_lh.append(llhh(lin).metric.draw_sample())
            with ift.random.Context(42):
                samps_ham.append(hh(lin).metric.draw_sample())
        allclose(samps_lh)
        allclose(samps_ham)

        mini_results = []
        for ham in hams_for_sampling:
            with ift.random.Context(42):
                mini_results.append(mini(ift.SampledKLEnergy(pos, ham, 3, None, comm=MPI.COMM_SELF))[0].position)
        allclose(mini_results)
