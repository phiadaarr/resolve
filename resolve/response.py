# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np
from ducc0.wgridder import dirty2ms, ms2dirty

import nifty7 as ift

from .global_config import epsilon, nthreads, wgridding
from .observation import Observation
from .util import my_assert, my_assert_isinstance, my_asserteq


def StokesIResponse(observation, domain):
    my_assert_isinstance(observation, Observation)
    domain = ift.DomainTuple.make(domain)
    my_asserteq(len(domain), 1)
    my_assert_isinstance(domain[0], ift.RGSpace)
    npol = observation.npol
    my_assert(npol in [1, 2])
    sp = observation.vis.dtype == np.complex64
    mask = observation.flags.val
    sr0 = SingleResponse(domain, observation.uvw, observation.freq, mask[0], sp)
    if npol == 1 or (npol == 2 and np.all(mask[0] == mask[1])):
        contr = ift.ContractionOperator(observation.vis.domain, 0)
        return contr.adjoint @ sr0
    elif npol == 2:
        sr1 = SingleResponse(domain, observation.uvw, observation.freq, mask[1], sp)
        return ResponseDistributor(sr0, sr1)
    raise RuntimeError


class ResponseDistributor(ift.LinearOperator):
    def __init__(self, *ops):
        dom, tgt = ops[0].domain, ops[0].target
        cap = self.TIMES | self.ADJOINT_TIMES
        for op in ops:
            my_assert_isinstance(op, ift.LinearOperator)
            my_assert(dom is op.domain)
            my_assert(tgt is op.target)
            my_assert(self.TIMES & op.capability, self.ADJOINT_TIMES & op.capability)
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
    def __init__(self, domain, uvw, freq, mask, single_precision):
        # TODO Currently only the response uses single_precision if possible. Could be rolled out to the whole likelihood
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.makeDomain(ift.UnstructuredDomain(ss) for ss in (uvw.shape[0], freq.size))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._args = {
            'uvw': uvw,
            'freq': freq,
            'mask': mask.astype(np.uint8),
            'nu': 0,
            'nv': 0,
            'pixsize_x': self._domain[0].distances[0],
            'pixsize_y': self._domain[0].distances[1],
            'epsilon': epsilon(),
            'do_wstacking': wgridding(),
            'nthreads': nthreads()
        }
        self._vol = self._domain[0].scalar_dvol
        self._target_dtype = np.complex64 if single_precision else np.complex128
        self._domain_dtype = np.float32 if single_precision else np.float64

    def apply(self, x, mode):
        self._check_input(x, mode)
        # my_asserteq(x.dtype, self._domain_dtype if mode == self.TIMES else self._target_dtype)
        x = x.val.astype(self._domain_dtype if mode == self.TIMES else self._target_dtype)
        if mode == self.TIMES:
            args1 = {'dirty': x}
            f = dirty2ms
        else:
            args1 = {
                'ms': x,
                'npix_x': self._domain[0].shape[0],
                'npix_y': self._domain.shape[1]
            }
            f = ms2dirty
        res = ift.makeField(self._tgt(mode), f(**self._args, **args1)*self._vol)
        my_asserteq(res.dtype, self._target_dtype if mode == self.TIMES else self._domain_dtype)
        return res
