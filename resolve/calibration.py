# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

from .util import my_assert, my_asserteq


class Calibration:
    def __init__(self, t_pix, t_max, antennas, xi_key, zero_padding_factor, amplitude, clip=[]):
        tdst = t_max/(t_pix - 1)
        tspace = ift.RGSpace(t_pix, distances=tdst)
        sp_ant = ift.UnstructuredDomain(len(antennas))
        dom = ift.DomainTuple.make((sp_ant, tspace))

        zp = ift.FieldZeroPadder(dom, (zero_padding_factor*tspace.shape[0],), space=1)
        self._op = zp.adjoint @ ht @ hop
        self.nozeropad = ht @ hop


def CalibrationDistributor(domain, ant, time):
    my_asserteq(len(time.shape), 1)
    my_asserteq(ant.shape, time.shape)
    my_assert(time.min() >= 0)
    my_assert(time.max() < domain[1].total_volume)
    dd = [ift.RGSpace(domain[0].shape, distances=1.), domain[1]]
    dd = ift.DomainTuple.make(dd)
    positions = np.array([ant, time])
    li = ift.LinearInterpolator(dd, positions)
    return li @ ift.Realizer(li.domain) @ ift.GeometryRemover(dd, 0).adjoint
