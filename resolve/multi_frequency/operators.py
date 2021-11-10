# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2021 Max-Planck-Society
# Authors: Philipp Frank, Philipp Arras, Philipp Haim

import nifty8 as ift
import numpy as np

from ..polarization_space import PolarizationSpace
from ..util import my_assert, my_assert_isinstance, my_asserteq
from .irg_space import IRGSpace


class WienerIntegrations(ift.LinearOperator):
    """Operator that performs the integrations necessary for an integrated
    Wiener process.

    Parameters
    ----------
    time_domain : IRGSpace
        Domain that contains the temporal information of the process.

    remaining_domain : DomainTuple or Domain
        All integrations are handled independently for this domain.
    """
    def __init__(self, time_domain, remaining_domain):
        my_assert_isinstance(time_domain, IRGSpace)
        self._target = ift.makeDomain((time_domain, remaining_domain))
        dom = ift.UnstructuredDomain((2, time_domain.size - 1)), remaining_domain
        self._domain = ift.makeDomain(dom)
        self._volumes = time_domain.dvol[:, None, None]
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        first, second = (0,), (1,)
        from_second = (slice(1, None),)
        no_border = (slice(0, -1),)
        reverse = (slice(None, None, -1),)
        if mode == self.TIMES:
            x = x.val
            res = np.zeros(self._target.shape)
            res[from_second] = np.cumsum(x[second], axis=0)
            res[from_second] = (
                res[from_second] + res[no_border]
            ) / 2 * self._volumes + x[first]
            res[from_second] = np.cumsum(res[from_second], axis=0)
        else:
            x = x.val_rw()
            res = np.zeros(self._domain.shape)
            x[from_second] = np.cumsum(x[from_second][reverse], axis=0)[reverse]
            res[first] += x[from_second]
            x[from_second] *= self._volumes / 2.0
            x[no_border] += x[from_second]
            res[second] += np.cumsum(x[from_second][reverse], axis=0)[reverse]
        return ift.makeField(self._tgt(mode), res)


def IntWProcessInitialConditions(a0, b0, wpop, irg_space=None):
    for op in [a0, b0]:
        my_assert(ift.is_operator(op))

    if ift.is_operator(wpop):
        tgt = wpop.target
    else:
        tgt = irg_space, a0.target[0]
    my_asserteq(a0.target[0], b0.target[0], tgt[1])

    sdom = tgt[0]

    bc = _FancyBroadcast(tgt)
    factors = ift.full(sdom, 0)
    factors = np.empty(sdom.shape)
    factors[0] = 0
    factors[1:] = np.cumsum(sdom.dvol)
    factors = ift.makeField(sdom, factors)
    res = bc @ a0 + ift.DiagonalOperator(factors, tgt, 0) @ bc @ b0

    if wpop is None:
        return res
    else:
        return wpop + res


class _FancyBroadcast(ift.LinearOperator):
    def __init__(self, target):
        my_asserteq(len(target), 2)
        my_asserteq(len(target[0].shape), 1)
        self._target = ift.DomainTuple.make(target)
        self._domain = ift.DomainTuple.make(target[1])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.broadcast_to(x.val[None], self._target.shape)
        else:
            res = np.sum(x.val, axis=0)
        return ift.makeField(self._tgt(mode), res)


class MfWeightingInterpolation(ift.LinearOperator):
    def __init__(self, eff_uvw, domain):
        domain = ift.DomainTuple.make(domain)
        my_asserteq(domain.shape[0], eff_uvw.shape[2])  # freqaxis
        self._domain = domain
        nrow, nfreq = eff_uvw.shape[1:]
        tgt = [PolarizationSpace("I")] + [ift.UnstructuredDomain(aa) for aa in [nrow, nfreq]]
        self._target = ift.DomainTuple.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        # FIXME Try to unify all those operators which loop over freq dimension
        self._ops = []
        for ii in range(nfreq):
            op = ift.LinearInterpolator(domain[1], eff_uvw.val[:, :, ii])
            self._ops.append(op)
        my_asserteq(self.target.shape[0], 1)

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = np.empty(self._tgt(mode).shape)
        if mode == self.TIMES:
            for ii, op in enumerate(self._ops):
                res[0, :, ii] = op(ift.makeField(op.domain, x.val[ii])).val
        else:
            for ii, op in enumerate(self._ops):
                op = op.adjoint
                res[ii] = op(ift.makeField(op.domain, x.val[0, :, ii])).val
        return ift.makeField(self._tgt(mode), res)
