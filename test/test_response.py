# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

from os.path import join

import nifty8 as ift
import numpy as np
import pytest

import resolve as rve

from .common import setup_function, teardown_function

pmp = pytest.mark.parametrize
np.seterr(all="raise")

direc = "/data/"
OBS = []
for polmode in ["all", "stokesi", "stokesiavg"]:
    oo = rve.ms2observations(
            f"{direc}CYG-ALL-2052-2MHZ.ms", "DATA", True, 0, polarizations=polmode
        )[0]
    # OBS.append(oo.to_single_precision())
    OBS.append(oo.to_double_precision())
npix, fov = 256, 1 * rve.DEG2RAD
sdom = ift.RGSpace((npix, npix), (fov / npix, fov / npix))
dom = rve.default_sky_domain(sdom=sdom, fdom=rve.IRGSpace([np.mean(OBS[0].freq)]))
FACETS = [(1, 1), (2, 2), (2, 1), (1, 4)]


@pmp("obs", OBS)
@pmp("facets", FACETS)
def test_single_response(obs, facets):
    obs = obs.to_double_precision()
    sdom = dom[-1]
    mask = obs.mask.val
    op = rve.SingleResponse(sdom, obs.uvw, obs.freq, mask[0], facets=facets)
    ift.extra.check_linear_operator(op, np.float64, np.complex128,
                                    only_r_linear=True, rtol=1e-6, atol=1e-6)


def test_facet_consistency():
    sdom = dom[-1]
    obs = OBS[0].to_double_precision()
    res0 = None
    pos = ift.from_random(sdom)
    for facets in FACETS:
        op = rve.SingleResponse(sdom, obs.uvw, obs.freq, obs.mask.val[0], facets=facets)
        res = op(pos)
        if res0 is None:
            res0 = res
        ift.extra.assert_allclose(res0, res, rtol=1e-4, atol=1e-4)
