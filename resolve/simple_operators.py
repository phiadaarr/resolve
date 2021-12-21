# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import numpy as np

import nifty8 as ift

from .util import my_assert, my_assert_isinstance, my_asserteq


class AddEmptyDimension(ift.LinearOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        my_asserteq(len(self._domain), 1)
        my_asserteq(len(self._domain.shape), 1)
        my_assert_isinstance(self._domain[0], ift.UnstructuredDomain)
        tmp = ift.UnstructuredDomain(1)
        self._target = ift.makeDomain(
            (tmp, ift.UnstructuredDomain((self._domain.shape[0])), tmp)
        )
        self._capability = self._all_ops

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode in [self.TIMES, self.ADJOINT_INVERSE_TIMES]:
            x = x.val[None, :, None]
        else:
            x = x.val[0, :, 0]
        return ift.makeField(self._tgt(mode), x)


class LinearOperatorOverAxis(ift.LinearOperator):
    def __init__(self, operator, domain):
        self._domain = ift.makeDomain(domain)
        assert isinstance(self._domain[0], ift.UnstructuredDomain)
        assert len(self._domain[0].shape) == 1
        n = self._domain.shape[0]
        assert ift.makeDomain(self._domain[1:]) == operator.domain
        tgt = [ift.UnstructuredDomain(n)] + list(operator.target)
        self._target = ift.makeDomain(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._op = operator

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = np.empty(self._tgt(mode).shape)
        op = self._op if mode == self.TIMES else self._op.adjoint
        for ii in range(x.shape[0]):
            res[ii] = op(ift.makeField(op.domain, x.val[ii])).val
        return ift.makeField(self._tgt(mode), res)


class AddEmptyDimensionAtEnd(ift.LinearOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(list(self._domain) + [ift.UnstructuredDomain(1)])
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            x = x.val[..., None]
        else:
            x = x.val[..., 0]
        return ift.makeField(self._tgt(mode), x)


def MultiDomainVariableCovarianceGaussianEnergy(data, signal_response, invcov):
    from .likelihood import get_mask_multi_field

    my_asserteq(data.domain, signal_response.target, invcov.target)
    my_assert_isinstance(data.domain, ift.MultiDomain)
    my_assert_isinstance(signal_response.domain, ift.MultiDomain)
    my_assert(ift.is_operator(invcov))
    my_assert(ift.is_operator(signal_response))
    res = []
    invcovfld = invcov(ift.full(invcov.domain, 1.0))
    mask = get_mask_multi_field(invcovfld)
    data = mask(data)
    signal_response = mask @ signal_response
    invcov = mask @ invcov
    for kk in data.keys():
        res.append(
            ift.VariableCovarianceGaussianEnergy(
                data.domain[kk], "resi" + kk, "icov" + kk, data[kk].dtype
            )
        )
    resi = ift.PrependKey(data.domain, "resi") @ ift.Adder(data, True) @ signal_response
    invcov = ift.PrependKey(data.domain, "icov") @ invcov
    return reduce(add, res) @ (resi + invcov)


class DomainChangerAndReshaper(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        assert self._domain.size == self._target.size

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        tgt = self._tgt(mode)
        return ift.makeField(tgt, x.reshape(tgt.shape))


class MultiFieldStacker(ift.LinearOperator):
    def __init__(self, target, space, keys):
        """
        Parameters
        ----------
        target : DomainTuple
            Target of the operator.
        space : int
            Index behind which the iterated domain shall be in the target
            domain. target[space] needs to be one-dimensional.
        keys : list
            List of keys. Needs to have same length as target[space].
        """
        self._target = ift.DomainTuple.make(target)
        dom = {kk: self._target[:space] + self._target[space + 1:] for kk in keys}
        self._domain = ift.makeDomain(dom)
        self._capability = self._all_ops
        self._keys = tuple(keys)
        self._space = int(space)
        assert self.domain.size == self.target.size

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES or mode == self.ADJOINT_INVERSE_TIMES:
            dtype = x[self._keys[0]].dtype
            for kk in self._keys:
                assert x[kk].dtype == dtype
            res = np.empty(self._target.shape, dtype=dtype)
            for ii, kk in enumerate(self._keys):
                res[self._slice(ii)] = x[kk]
        else:
            res = {kk: x[self._slice(ii)] for ii, kk in enumerate(self._keys)}
        return ift.makeField(self._tgt(mode), res)

    def _slice(self, index):
        ndim = reduce(add, (len(self._target[ii].shape) for ii in range(self._space)), 0)
        return ndim * (slice(None),) + (index,)
