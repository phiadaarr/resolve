# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty8 as ift

from .data.observation import Observation
from .util import my_assert, my_assert_isinstance, my_asserteq


def calibration_distribution(
    observation, phase_operator, logamplitude_operator, antenna_dct, time_dct=None
):
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(phase_operator, logamplitude_operator, ift.Operator)
    dom = phase_operator.target
    my_asserteq(dom, logamplitude_operator.target)
    tgt = observation.vis.domain
    ap = observation.antenna_positions
    cop1 = CalibrationDistributor(dom, tgt, ap.ant1, ap.time, antenna_dct, time_dct)
    cop2 = CalibrationDistributor(dom, tgt, ap.ant2, ap.time, antenna_dct, time_dct)
    res0 = (cop1 + cop2).real @ logamplitude_operator
    res1 = (1j * (cop1 - cop2).real) @ phase_operator
    return (res0 + res1).exp()


class CalibrationDistributor(ift.LinearOperator):
    def __init__(self, domain, target, ant_col, time_col, antenna_dct, time_dct):
        # Domain: pols, antennas, time, freqs
        # Target: pols, times, freqs
        my_asserteq(ant_col.ndim, 1)
        my_assert(np.issubdtype(ant_col.dtype, np.integer))
        my_assert(set(np.unique(ant_col)) <= set(antenna_dct.keys()))
        my_assert(np.min(ant_col) >= 0)
        my_assert(np.max(ant_col) < len(antenna_dct))
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._nantennas = len(antenna_dct)

        # Support missing antennas
        myants = np.empty_like(ant_col)
        for kk, vv in antenna_dct.items():
            myants[ant_col == kk] = vv
        tdom = self._domain[2]
        if isinstance(tdom, ift.RGSpace):
            my_assert(time_dct is None)
            mytime = time_col
            self._ntimes = tdom.size
        else:
            mytime = np.empty(time_col.size, int)
            for kk, vv in time_dct.items():
                mytime[time_col == kk] = vv
            self._ntimes = len(time_dct)
        self._li = MyLinearInterpolator(tdom, self._nantennas, myants, mytime)

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = None
        x = x.val
        npol, _, nfreq = self._target.shape
        for pol in range(npol):
            for freq in range(nfreq):
                op = self._li if mode == self.TIMES else self._li.adjoint
                val = x[pol, :, :, freq] if mode == self.TIMES else x[pol, :, freq]
                tmp = op(ift.makeField(op.domain, val)).val
                if res is None:
                    shp = self._target.shape
                    if mode == self.ADJOINT_TIMES:
                        shp = (
                            self._target.shape[0],
                            self._nantennas,
                            self._ntimes,
                            self._target.shape[2],
                        )
                    res = np.empty(shp, dtype=tmp.dtype)
                if mode == self.TIMES:
                    res[pol, :, freq] = tmp
                else:
                    res[pol, :, :, freq] = tmp
        return ift.makeField(self._tgt(mode), res)


class MyLinearInterpolator(ift.LinearOperator):
    def __init__(self, time_domain, nants, ant_col, time_col):
        my_assert_isinstance(time_domain, (ift.RGSpace, ift.UnstructuredDomain))
        my_asserteq(len(time_domain.shape), 1)
        floattime = isinstance(time_domain, ift.RGSpace)
        dt = time_domain.distances[0] if floattime else 1
        dom = ift.DomainTuple.make((ift.RGSpace((nants, time_domain.size), (1, dt))))
        self._domain = ift.makeDomain((ift.UnstructuredDomain(nants), time_domain))
        self._li = ift.LinearInterpolator(dom, np.array([ant_col, time_col]))
        self._target = self._li.target
        self._capability = self.TIMES | self.ADJOINT_TIMES
        my_asserteq(ant_col.ndim, time_col.ndim, 1)
        my_asserteq(ant_col.shape, time_col.shape)
        my_assert(np.min(ant_col) >= 0)
        my_assert(np.max(ant_col) < self._domain.shape[0] * dom[0].distances[0])
        my_assert(np.min(time_col) >= 0)
        my_assert(np.max(time_col) < self._domain.shape[1] * dom[0].distances[1])
        my_assert(np.issubdtype(ant_col.dtype, np.integer))
        my_assert(
            np.issubdtype(time_col.dtype, np.floating if floattime else np.integer)
        )

    def apply(self, x, mode):
        self._check_input(x, mode)
        op = self._li if mode == self.TIMES else self._li.adjoint
        return op(ift.makeField(op.domain, x.val))
