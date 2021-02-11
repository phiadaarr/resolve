import nifty7 as ift
import numpy as np


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


class MPIAdder(ift.EndomorphicOperator):
    def __init__(self, domain, comm, _callfrommake=False):
        if not _callfrommake:
            raise RuntimeError("Use MPIAdder.make for instantiation.")
        self._domain = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._comm = comm

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = ift.utilities.allreduce_sum([x.val], self._comm)
        else:
            # FIXME assert x same on all tasks?
            raise NotImplementedError("Try mpi.gather here.")
            res = x.val
        return ift.makeField(self._tgt(mode), res)

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

    # Instantiate response (dependent on rank via e.g. freq)
    # In the end: MfResponse(...., freq[lo:hi], ...) @ SkySlicer
    R = DummyResponse(domain, d.domain)
    # Warning: the next line changes the random state!
    # ift.extra.check_linear_operator(R)
    localop = ift.GaussianEnergy(d) @ R
    adder = MPIAdder.make(localop.target, comm)
    # ift.extra.check_linear_operator(adder)
    totalop = adder @ localop
    return totalop


def main():
    ddomain = ift.UnstructuredDomain(4), ift.UnstructuredDomain(1)
    comm, size, rank, master = ift.utilities.get_MPI_params()
    data = ift.from_random(ddomain)
    if master:
        np.save("data.npy", data.val)
    comm.Barrier()

    sky_domain = ift.RGSpace((2, 2))

    lh = getop(sky_domain, comm)
    lh1 = getop(sky_domain, None)
    pos = ift.from_random(lh.domain)
    res0 = lh(pos)
    res1 = lh1(pos)
    ift.extra.assert_allclose(res0, res1)


if __name__ == "__main__":
    main()
