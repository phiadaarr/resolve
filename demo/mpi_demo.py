import nifty7 as ift
import numpy as np


class DummyResponse(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if self.TIMES:
            res = np.full(self._tgt(mode).shape, np.sum(x.val))
        else:
            res = np.full(self._tgt(mode).shape, np.sum(x.val))
        return ift.makeField(self._tgt(mode), res)


class MPIAdder(ift.EndomorphicOperator):
    def __init__(self, domain, comm):
        self._domain = ift.makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._comm = comm

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = ift.utilities.allreduce_sum([x.val], self._comm)
        else:
            # FIXME assert x same on all tasks?
            res = x.val
        return ift.makeField(self._tgt(mode), res)


def getop(domain, comm=None):
    """Return energy operator that maps the full multi-frequency sky onto
    the log-likelihood value for a frequency slice."""

    if comm is None:
        # Use default MPI communicator
        comm, size, rank, master = ift.utilities.get_MPI_params()
    elif not comm:
        # Do not parallelize (intended for testing)
        size, rank, master = ift.utilities.get_MPI_params_from_comm(None)
    else:
        # Use custom communicator
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
    ift.extra.check_linear_operator(R)
    localop = ift.GaussianEnergy(d) @ R
    adder = MPIAdder(localop.target, comm)
    # ift.extra.check_linear_operator(adder)
    totalop = adder @ localop
    return totalop


def main():
    ddomain = ift.UnstructuredDomain(10), ift.UnstructuredDomain(100)
    data = ift.from_random(ddomain)
    np.save("data.npy", data.val)
    sky_domain = ift.RGSpace((2, 2))
    lh = getop(sky_domain)
    lh1 = getop(sky_domain, False)
    pos = ift.from_random(lh.domain)
    print(lh(pos))
    print(lh1(pos))


if __name__ == "__main__":
    # _, size, rank, master = ift.utilities.get_MPI_params()
    # print(ift.utilities.shareRange(100, size, rank))

    main()
