# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
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
    mask = ift.makeField(obs.mask.domain[1:], obs.mask.val[0])
    op = rve.SingleResponse(sdom, obs.uvw, obs.freq, mask=mask, facets=facets, epsilon=1e-6,
                            do_wgridding=False)
    ift.extra.check_linear_operator(op, np.float64, np.complex128,
                                    only_r_linear=True, rtol=1e-6, atol=1e-6)


def test_facet_consistency():
    sdom = dom[-1]
    obs = OBS[0].to_double_precision()
    res0 = None
    pos = ift.from_random(sdom)
    for facets in FACETS:
        mask = ift.makeField(obs.mask.domain[1:], obs.mask.val[0])
        op = rve.SingleResponse(sdom, obs.uvw, obs.freq, False, 1e-6, facets=facets)
        res = op(pos)
        if res0 is None:
            res0 = res
        ift.extra.assert_allclose(res0, res, rtol=1e-4, atol=1e-4)
