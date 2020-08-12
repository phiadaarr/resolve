# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np
from ducc0.wgridder import dirty2ms, ms2dirty

import nifty7 as ift

from .global_config import epsilon, nthreads, wstacking
from .util import (complex2float_dtype, my_assert, my_assert_isinstance,
                   my_asserteq)

# FIXME Use flagging functionality from ducc0


def StokesIResponse(observation, domain):
    npol = observation.vis.shape[0]
    my_assert(npol in [1, 2])
    mask = (observation.weight.val > 0).astype(complex2float_dtype(observation.vis.dtype))
    sr0 = SingleResponse(domain, observation.uvw, observation.freq, mask[0])
    if npol == 1 or (npol == 2 and np.all(mask[0] == mask[1])):
        contr = ift.ContractionOperator(observation.vis.domain, 0)
        return contr.adjoint @ sr0
    elif npol == 2:
        sr1 = SingleResponse(domain, observation.uvw, observation.freq, mask[1])
        return ResponseDistributor(sr0, sr1)


class ResponseDistributor(ift.LinearOperator):
    def __init__(self, *ops):
        dom, tgt = ops[0].domain, ops[0].target
        cap = self.TIMES | self.ADJOINT_TIMES
        for op in ops:
            my_assert_isinstance(op, ift.LinearOperator)
            my_assert(dom is op.domain)
            my_assert(tgt is op.target)
            my_asserteq(cap, op.capability)
        self._domain = ift.makeDomain(dom)
        self._target = ift.makeDomain((ift.UnstructuredDomain(len(ops)), *tgt))
        self._capability = cap
        self._ops = ops

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = []
            for op in self._ops:
                res.append(op(x).val)
            res = np.array(res)
            return ift.makeField(self._tgt(mode), np.array(res))
        for ii, op in enumerate(self._ops):
            new = op.adjoint(ift.makeField(self._ops[0].target, x.val[ii]))
            if ii == 0:
                res = new
            else:
                res = res + new
        return res


class FullResponse(ift.LinearOperator):
    def __init__(self, observation, sky_domain):
        raise NotImplementedError


class SingleResponse(ift.LinearOperator):
    def __init__(self, domain, uvw, freq, mask):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.makeDomain(ift.UnstructuredDomain(ss) for ss in (uvw.shape[0], freq.size))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._args = {
            'uvw': uvw,
            'freq': freq,
            'wgt': mask,
            'nu': 0,
            'nv': 0,
            'pixsize_x': self._domain[0].distances[0],
            'pixsize_y': self._domain[0].distances[1],
            'epsilon': epsilon(),
            'do_wstacking': wstacking(),
            'nthreads': nthreads()
        }
        self._vol = self._domain[0].scalar_dvol

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            args1 = {'dirty': x.val.astype(self._args['wgt'].dtype)}
            f = dirty2ms
        else:
            args1 = {
                'ms': x.val,
                'npix_x': self._domain[0].shape[0],
                'npix_y': self._domain.shape[1]
            }
            f = ms2dirty
        return ift.makeField(self._tgt(mode), f(**self._args, **args1)*self._vol)
