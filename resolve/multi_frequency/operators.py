# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2013-2020 Max-Planck-Society
# Authors: Philipp Frank, Philipp Arras, Philipp Haim

import numpy as np

import nifty7 as ift


class WienerIntegrations(ift.LinearOperator):
    def __init__(self, freqdomain, imagedomain):
        # FIXME Write interface checks
        self._target = ift.makeDomain((freqdomain, imagedomain))
        dom = list(self._target)
        dom = ift.UnstructuredDomain((2, freqdomain.size-1)), imagedomain
        self._domain = ift.makeDomain(dom)
        self._volumes = freqdomain.dvol[:-1][:, None, None]  # FIXME
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
