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


class _MPIAdder(ift.EndomorphicOperator):
    def __init__(self, domain, comm, _callfrommake=False):
        if not _callfrommake:
            raise RuntimeError("Use MPIAdder.make for instantiation.")
        self._domain = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._comm = comm

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            res = allreduce_sum([x.val], self._comm)
            return ift.makeField(self.target, res)

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
        return x

    @classmethod
    def make(cls, domain, comm):
        if comm is None:
            return ift.ScalingOperator(domain, 1.0)
        return cls(domain, comm, True)


class MPIAdder(ift.Operator):
    def __init__(self, domain, comm, _callfrommake=False):
        if not _callfrommake:
            raise RuntimeError("Use MPIAdder.make for instantiation.")
        self._domain = self._target = ift.makeDomain(domain)
        self._comm = comm

    def apply(self, x):
        self._check_input(x)
        if not ift.is_linearization(x):
            return allreduce_sum([x], self._comm)
        val = allreduce_sum([x.val], self._comm)
        # FIXME Why normalizing?
        norm = 1 / self._comm.Get_size()
        jac = allreduce_sum([x.jac], self._comm).scale(norm)
        return x.new(val, jac)

    def __call__(self, x):
        # Hook for preserving the metric
        if x.metric is not None:
            metric = allreduce_sum([x.metric], self._comm)
            return self.apply(x.trivial_jac()).prepend_jac(x.jac).add_metric(metric)
        return super(MPIAdder, self).__call__(x)

    @classmethod
    def make(cls, domain, comm):
        if comm is None:
            return ift.ScalingOperator(domain, 1.0)
        return cls(domain, comm, True)


def getop(domain, comm):
    """Return energy operator that maps the full multi-frequency sky onto
    the log-likelihood value for a frequency slice."""
    size, rank, master = ift.utilities.get_MPI_params_from_comm(comm)

    # Load data
    d = np.load("data.npy")

    # Select relevant part of data
    nwork = d.shape[0]
    lo, hi = ift.utilities.shareRange(nwork, size, rank)
    d = d[lo:hi]
    d = ift.makeField((ift.UnstructuredDomain(ii) for ii in d.shape), d)

    # Instantiate response and likelihood (dependent on rank via e.g. freq)
    R = DummyResponse(domain, d.domain)
    localop = ift.GaussianEnergy(d) @ R

    # Add contributions from all tasks together
    adder = MPIAdder.make(localop.target, comm)
    return adder @ localop


def main():
    ddomain = ift.UnstructuredDomain(4), ift.UnstructuredDomain(1)
    comm, size, rank, master = ift.utilities.get_MPI_params()
    data = ift.from_random(ddomain)
    if master:
        np.save("data.npy", data.val)
    if comm is not None:
        comm.Barrier()

    sky_domain = ift.RGSpace((2, 2))
    lh = getop(sky_domain, comm)
    lh1 = getop(sky_domain, None)

    pos = ift.from_random(lh.domain)
    res0 = lh(pos)
    res1 = lh1(pos)
    ift.extra.assert_allclose(res0, res1)

    for wm in [False, True]:
        lin = ift.Linearization.make_var(pos, wm)
        res0 = lh(lin)
        res1 = lh1(lin)
        ift.extra.assert_allclose(res0.val, res1.val)
        for _ in range(10):
            foo = ift.from_random(lh.domain)
            # FIXME Is this really what we want?
            ift.extra.assert_allclose(
                allreduce_sum([res0.jac(foo)], comm), res1.jac(foo)
            )
        # FIXME Is this really what we want?
        grad0 = allreduce_sum([res0.gradient], comm)
        ift.extra.assert_allclose(grad0, res1.gradient)
        if not wm:
            continue

        foo = ift.from_random(lh.domain)
        ift.extra.assert_allclose(res0.metric(foo), res1.metric(foo))


if __name__ == "__main__":
    main()
