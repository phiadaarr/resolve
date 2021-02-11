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
    return ift.GaussianEnergy(d) @ R


def main():
    ddomain = ift.UnstructuredDomain(10), ift.UnstructuredDomain(100)
    data = ift.from_random(ddomain)
    np.save("data.npy", data.val)
    sky_domain = ift.RGSpace((2, 2))
    lh = getop(sky_domain)

    # TODO MPI allreduce or something similar


if __name__ == "__main__":
    # _, size, rank, master = ift.utilities.get_MPI_params()
    # print(ift.utilities.shareRange(100, size, rank))

    main()
