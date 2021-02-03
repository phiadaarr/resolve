# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

from .util import my_assert_isinstance, my_asserteq


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
