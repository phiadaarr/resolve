import nifty7 as ift
import numpy as np
from nifty7.utilities import allreduce_sum


class DummyResponse(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.full(self._tgt(mode).shape, np.sum(x.val))
        else:
            res = np.full(self._tgt(mode).shape, np.sum(x.val))
        return ift.makeField(self._tgt(mode), res)

    def __repr__(self):
        return f"DummyResponse {self._domain.shape} -> {self._target.shape}"


# def MPISamplingDistributor(op, comm):
#     f = op.draw_sample_with_dtype

#     def newfunc(dtype, from_inverse=False):
#         ntask, rank, _ = ift.utilities.get_MPI_params_from_comm(comm)
#         sseq = ift.random.spawn_sseq(ntask)
#         with ift.random.Context(sseq[rank]):
#             res = f(dtype, from_inverse)
#         return res

#     op.draw_sample_with_dtype = newfunc
#     return op


def getop(domain, comm):
    """Return energy operator that maps the full multi-frequency sky onto
    the log-likelihood value for a frequency slice."""

    # Non-parallel operators
    sky = ift.FieldAdapter(domain, "sky").exp()

    def get_local_oplist(comm):
        d = np.load("data.npy")
        invcov = np.load("invcov.npy")
        nwork = d.shape[0]
        size, rank, _ = ift.utilities.get_MPI_params_from_comm(comm)
        local_indices = range(*ift.utilities.shareRange(nwork, size, rank))
        lst = []
        for ii in local_indices:
            ddom = ift.UnstructuredDomain(d[ii].shape)
            dd = ift.makeField(ddom, d[ii])
            iicc = ift.makeOp(ift.makeField(ddom, invcov[ii]))
            rr = DummyResponse(domain, ddom)
            ee = ift.GaussianEnergy(dd, iicc)
            lst.append(ee @ rr)
        return lst

    return AllreduceSum(get_local_oplist, comm) @ sky


class AllreduceSum(ift.Operator):
    def __init__(self, get_oplist, comm):
        self._comm = comm
        self._oplist = get_oplist(comm)
        self._domain = self._oplist[0].domain
        self._target = self._oplist[0].target

    def apply(self, x):
        self._check_input(x)
        return ift.utilities.allreduce_sum([op(x) for op in self._oplist], self._comm)


def main():
    ddomain = ift.UnstructuredDomain(7), ift.UnstructuredDomain(1)
    comm, size, rank, master = ift.utilities.get_MPI_params()
    data = ift.from_random(ddomain).exp()
    invcov = ift.from_random(ddomain).exp() * 0 + 1
    if master:
        np.save("data.npy", data.val)
        np.save("invcov.npy", invcov.val)
    if comm is not None:
        comm.Barrier()

    sky_domain = ift.RGSpace((100, 100))
    lh = getop(sky_domain, comm)
    lh1 = getop(sky_domain, None)

    # Evaluate Field
    pos = ift.from_random(lh.domain)
    res0 = lh(pos)
    res1 = lh1(pos)
    ift.extra.assert_allclose(res0, res1)

    # Evaluate Linearization
    for wm in [False, True]:
        lin = ift.Linearization.make_var(pos, wm)
        res0 = lh(lin)
        res1 = lh1(lin)
        ift.extra.assert_allclose(res0.val, res1.val)
        for _ in range(10):
            foo = ift.from_random(lh.domain)
            ift.extra.assert_allclose(res0.jac(foo), res1.jac(foo))

            foo = ift.from_random(res0.jac.target)
            ift.extra.assert_allclose(res0.jac.adjoint(foo), res1.jac.adjoint(foo))
        ift.extra.assert_allclose(res0.gradient, res1.gradient)

        # Dummy minimization
        pos = ift.from_random(lh.domain)
        ham = ift.StandardHamiltonian(
            lh, ift.GradientNormController(iteration_limit=10)
        )
        ham1 = ift.StandardHamiltonian(
            lh1, ift.GradientNormController(iteration_limit=10)
        )
        e = ift.EnergyAdapter(pos, ham, want_metric=wm)
        e1 = ift.EnergyAdapter(pos, ham1, want_metric=wm)

        mini = ift.NewtonCG if wm else ift.SteepestDescent
        mini = mini(ift.GradientNormController(iteration_limit=5))
        mini_e, _ = mini(e)
        mini_e1, _ = mini(e1)
        ift.extra.assert_allclose(mini_e.position, mini_e1.position)

        if not wm:
            continue

        foo = ift.from_random(lh.domain)
        ift.extra.assert_allclose(res0.metric(foo), res1.metric(foo))

        # Draw samples
        pos = ift.from_random(lh.domain)
        lin = ift.Linearization.make_var(pos, True)
        samp = ham(lin).metric.draw_sample()
        samp1 = ham1(lin).metric.draw_sample()
        # ift.extra.assert_allclose(samp, samp1)


if __name__ == "__main__":
    main()
