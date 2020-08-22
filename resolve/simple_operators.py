# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift
from .util import my_assert_isinstance, my_asserteq


class AddEmptyDimension(ift.LinearOperator):
    def __init__(self, domain):
        self._domain = ift.makeDomain(domain)
        my_asserteq(len(self._domain), 1)
        my_asserteq(len(self._domain.shape), 1)
        my_assert_isinstance(self._domain[0], ift.UnstructuredDomain)
        tmp = ift.UnstructuredDomain(1)
        self._target = ift.makeDomain((tmp, ift.UnstructuredDomain((self._domain.shape[0])), tmp))
        self._capability = self._all_ops

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode in [self.TIMES, self.ADJOINT_INVERSE_TIMES]:
            x = x.val[None, :, None]
        else:
            x = x.val[0, :, 0]
        return ift.makeField(self._tgt(mode), x)
