# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np
from ducc0.wgridder import dirty2ms, ms2dirty

import nifty7 as ift

from .util import complex2float_dtype, my_assert


def StokesIResponse(observation, domain, nthreads, epsilon, wstacking):
    npol = observation.vis.shape[0]
    my_assert(npol in [1, 2])
    mask = (observation.weight > 0).astype(complex2float_dtype(observation.vis.dtype))
    if npol == 1 or (npol == 2 and np.all(mask[0] == mask[1])):
        sr = SingleResponse(domain, observation.uvw, observation.freq, mask[0],
                            nthreads, epsilon, wstacking)
    elif npol == 2:
        raise NotImplementedError
    contr = ift.ContractionOperator((ift.UnstructuredDomain(npol), sr.target[0]), 0)
    return contr.adjoint @ sr


class FullResponse(ift.LinearOperator):
    def __init__(self, observation, sky_domain, nthreads, epsilon, wstacking):
        raise NotImplementedError


class SingleResponse(ift.LinearOperator):
    def __init__(self, domain, uvw, freq, mask, nthreads, epsilon, wstacking):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.makeDomain(ift.UnstructuredDomain((uvw.shape[0], freq.size)))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._args = {
            'uvw': uvw,
            'freq': freq,
            'wgt': mask,
            'nu': 0,
            'nv': 0,
            'pixsize_x': self._domain[0].distances[0],
            'pixsize_y': self._domain[0].distances[1],
            'epsilon': epsilon,
            'do_wstacking': wstacking,
            'nthreads': nthreads
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
