# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# Authors: Philipp Frank, Philipp Arras, Philipp Haim

import numpy as np

import nifty7 as ift

from ..util import my_assert_isinstance, my_asserteq


class WienerIntegrations(ift.LinearOperator):
    def __init__(self, freqdomain, imagedomain):
        # FIXME Write interface checks
        self._target = ift.makeDomain((freqdomain, imagedomain))
        dom = list(self._target)
        dom = ift.UnstructuredDomain((2, freqdomain.size-1)), imagedomain
        self._domain = ift.makeDomain(dom)
        self._volumes = freqdomain.dvol[:, None, None]
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        # FIXME If it turns out that this operator is a performance
        # bottleneck we can try implement it in parallel in C++. But
        # it may be hard to achieve good scaling because I think it
        # becomes memory bound quickly.
        self._check_input(x, mode)
        first, second = (0,), (1,)
        from_second = (slice(1, None),)
        no_border = (slice(0, -1),)
        reverse = (slice(None, None, -1),)
        if mode == self.TIMES:
            x = x.val
            res = np.zeros(self._target.shape)
            res[from_second] = np.cumsum(x[second], axis=0)
            res[from_second] = (res[from_second] + res[no_border])/2*self._volumes + x[first]
            res[from_second] = np.cumsum(res[from_second], axis=0)
        else:
            x = x.val_rw()
            res = np.zeros(self._domain.shape)
            x[from_second] = np.cumsum(x[from_second][reverse], axis=0)[reverse]
            res[first] += x[from_second]
            x[from_second] *= self._volumes/2.
            x[no_border] += x[from_second]
            res[second] += np.cumsum(x[from_second][reverse], axis=0)[reverse]
        return ift.makeField(self._tgt(mode), res)


def IntWProcessInitialConditions(a0, b0, wfop):
    for op in [a0, b0, wfop]:
        ift.is_operator(op)
    my_asserteq(a0.target, b0.target, ift.makeDomain(wfop.target[1]))
    bc = _FancyBroadcast(wfop.target)
    factors = ift.full(wfop.target[0], 0)
    factors = np.empty(wfop.target[0].shape)
    factors[0] = 0
    factors[1:] = np.cumsum(wfop.target[0].dvol)
    factors = ift.makeField(wfop.target[0], factors)
    return wfop + bc @ a0 + ift.DiagonalOperator(factors, wfop.target, 0) @ bc @ b0


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
        my_asserteq(domain.shape[0], eff_uvw.shape[1])  # freqaxis
        self._domain = domain
        nrow, nfreq = eff_uvw.shape
        tgt = [ift.UnstructuredDomain(aa) for aa in [1, nrow, nfreq]]
        self._target = ift.DomainTuple.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        # FIXME Try to unify all those operators which loop over freq dimension
        self._ops = []
        for ii in range(nfreq):
            op = ift.LinearInterpolator(domain[1], eff_uvw[:, ii][None])
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
