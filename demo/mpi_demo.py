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


class MPIAdder(ift.EndomorphicOperator):
    def __init__(self, domain, comm, _callfrommake=False):
        if not _callfrommake:
            raise RuntimeError("Use MPIAdder.make for instantiation.")
        self._domain = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._comm = comm

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.ADJOINT_TIMES:
            # FIXME Is this too expensive? If so, remove eventually.
            self._input_sanity_check(x)
            return x
        res = allreduce_sum([x.val], self._comm)
        return ift.makeField(self.target, res)

    def _input_sanity_check(self, x):
        if not isinstance(self._domain, ift.DomainTuple):
            raise NotImplementedError("Sanity check not implemented")
        size, rank, master = ift.utilities.get_MPI_params_from_comm(self._comm)
        recv = None
        if master:
            shp = (self._comm.Get_size(),) + self._domain.shape
            recv = np.empty(shp, dtype=x.val.dtype)
        self._comm.Gather(x.val, recv, root=0)
        if master:
            ref = recv[0]
            for ii in range(size):
                np.testing.assert_equal(recv[ii], ref)

    @classmethod
    def make(cls, domain, comm):
        if comm is None:
            return ift.ScalingOperator(domain, 1.0)
        return cls(domain, comm, True)

    def __call__(self, x):
        # Hook for preserving the metric
        if x.want_metric:
            return x.new(self(x._val), self).prepend_jac(x.jac).add_metric(x.metric)
        return super(MPIAdder, self).__call__(x)


def getop(domain, comm):
    """Return energy operator that maps the full multi-frequency sky onto
    the log-likelihood value for a frequency slice."""

    # Load data
    d = np.load("data.npy")

    if comm == -1:
        d = ift.makeField((ift.UnstructuredDomain(ii) for ii in d.shape), d)
    else:
        # Select relevant part of data
        size, rank, master = ift.utilities.get_MPI_params_from_comm(comm)
        nwork = d.shape[0]
        lo, hi = ift.utilities.shareRange(nwork, size, rank)
        d = d[lo:hi]
        d = ift.makeField((ift.UnstructuredDomain(ii) for ii in d.shape), d)

    # Instantiate response and likelihood (dependent on rank via e.g. freq)
    R = DummyResponse(domain, d.domain)
    sky = ift.FieldAdapter(R.domain, "sky").exp()
    localop = ift.GaussianEnergy(d) @ R

    if comm == -1:
        return localop @ sky

    # Add contributions from all tasks together
    adder = MPIAdder.make(localop.target, comm)
    return adder @ localop @ MPIAdder.make(R.domain, comm).adjoint @ sky


def main():
    ddomain = ift.UnstructuredDomain(4), ift.UnstructuredDomain(1)
    comm, size, rank, master = ift.utilities.get_MPI_params()
    data = ift.from_random(ddomain).exp()
    if master:
        np.save("data.npy", data.val)
    if comm is not None:
        comm.Barrier()

    sky_domain = ift.RGSpace((100, 100))
    lh = getop(sky_domain, comm)
    lh1 = getop(sky_domain, None)
    lh2 = getop(sky_domain, -1)

    # Evaluate Field
    pos = ift.from_random(lh.domain)
    res0 = lh(pos)
    res1 = lh1(pos)
    res2 = lh2(pos)
    ift.extra.assert_allclose(res0, res1)
    ift.extra.assert_allclose(res0, res2)

    # Evaluate Linearization
    for wm in [False, True]:
        lin = ift.Linearization.make_var(pos, wm)
        res0 = lh(lin)
        res1 = lh1(lin)
        res2 = lh2(lin)
        ift.extra.assert_allclose(res0.val, res1.val)
        ift.extra.assert_allclose(res0.val, res2.val)
        for _ in range(10):
            foo = ift.from_random(lh.domain)
            ift.extra.assert_allclose(res0.jac(foo), res1.jac(foo))
            ift.extra.assert_allclose(res0.jac(foo), res2.jac(foo))

            foo = ift.from_random(res0.jac.target)
            ift.extra.assert_allclose(res0.jac.adjoint(foo), res1.jac.adjoint(foo))
            ift.extra.assert_allclose(res0.jac.adjoint(foo), res2.jac.adjoint(foo))
        ift.extra.assert_allclose(res0.gradient, res1.gradient)
        ift.extra.assert_allclose(res0.gradient, res2.gradient)

        # Dummy minimization
        pos = ift.from_random(lh.domain)
        ham = ift.StandardHamiltonian(
            lh, ift.GradientNormController(iteration_limit=10)
        )
        ham1 = ift.StandardHamiltonian(
            lh1, ift.GradientNormController(iteration_limit=10)
        )
        ham2 = ift.StandardHamiltonian(
            lh2, ift.GradientNormController(iteration_limit=10)
        )
        e = ift.EnergyAdapter(pos, ham, want_metric=wm)
        e1 = ift.EnergyAdapter(pos, ham1, want_metric=wm)
        e2 = ift.EnergyAdapter(pos, ham2, want_metric=wm)

        mini = ift.NewtonCG if wm else ift.SteepestDescent
        mini = mini(ift.GradientNormController(iteration_limit=5))
        mini_e, _ = mini(e)
        mini_e1, _ = mini(e1)
        mini_e2, _ = mini(e2)
        ift.extra.assert_allclose(mini_e.position, mini_e1.position)
        ift.extra.assert_allclose(mini_e.position, mini_e2.position)

        if not wm:
            continue

        foo = ift.from_random(lh.domain)
        ift.extra.assert_allclose(res0.metric(foo), res1.metric(foo))

        # Draw samples
        # FIXME Write better test
        pos = ift.from_random(lh.domain)
        lin = ift.Linearization.make_var(pos, True)
        ham(lin).metric.draw_sample()
        ham1(lin).metric.draw_sample()
        ham2(lin).metric.draw_sample()


if __name__ == "__main__":
    main()
