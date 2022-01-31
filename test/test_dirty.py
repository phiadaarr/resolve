# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2022 Max-Planck-Society
# Author: Philipp Arras

from os.path import join

import numpy as np
import pytest

import nifty8 as ift
import resolve as rve

pmp = pytest.mark.parametrize
np.seterr(all="raise")

rve.set_epsilon(1e-9)
rve.set_wgridding(False)
rve.set_nthreads(8)


@pmp("with_zero_mode", [True])
@pmp("npol", [1, 2])
@pmp("flux_source", [1.])
@pmp("weights", [1.])
@pmp("weighting", ["natural", "uniform"])
def test_dirty(with_zero_mode, weights, flux_source, npol, weighting):
    npix = 100
    fov = "0.05deg"
    dst = rve.str2rad("0.1deg") / npix
    sdom = ift.RGSpace([npix, npix], [dst, dst])
    obs = _generate_testing_data(sdom, npol, weights=weights, flux_source=flux_source, with_zero_mode=with_zero_mode)
    dirty = rve.dirty_image(obs, weighting, fov, npix)
    psf = rve.dirty_image(obs, weighting, fov, npix, vis=ift.full(obs.vis.domain, 1.+0j))
    if with_zero_mode:
        rve.ubik_tools.field2fits(dirty, "dirty.fits", [obs])
        np.testing.assert_allclose(dirty.ducktape_left(sdom).s_integrate(), 1.)


def _generate_testing_data(sdom, npol, weights, flux_source, with_zero_mode):
    freq = np.array([1e9])
    f_over_c = freq/rve.SPEEDOFLIGHT
    hdst = sdom.get_default_codomain().distances
    npix_x, npix_y = sdom.shape
    xx, yy = (np.mgrid[:npix_x, :npix_y].astype(float) - 0.5*np.array(sdom.shape)[:, None, None]) / f_over_c
    xx = hdst[0]*xx.ravel()
    yy = hdst[1]*yy.ravel()
    if not with_zero_mode:
        ind = np.logical_and(xx == 0., yy == 0.)
        assert np.sum(ind) == 1.
        xx = xx[~ind]
        yy = yy[~ind]
    uvw = np.vstack([xx, yy, np.zeros_like(xx)]).T

    antpos = rve.AntennaPositions(uvw)
    if npol == 1:
        pol = rve.Polarization.trivial()
    else:
        pol = rve.Polarization((5, 8))
    vis = np.ones((npol, uvw.shape[0], 1), dtype=complex) * flux_source
    weight = np.ones(vis.shape, dtype=float)
    return rve.Observation(antpos, vis, weight, pol, freq)
