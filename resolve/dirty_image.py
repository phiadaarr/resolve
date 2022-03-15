# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2022 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np

from .logger import logger
from .constants import SPEEDOFLIGHT, str2rad
from .irg_space import IRGSpace
from .polarization_space import PolarizationSpace
from .response_new import InterferometryResponse
from .util import assert_sky_domain


def dirty_image(observation, weighting, sky_domain, do_wgridding, epsilon,
                vis=None, weight=None, nthreads=1):
    assert_sky_domain(sky_domain)
    pol = observation.polarization.has_crosshanded()
    if pol:
        raise NotImplementedError
    R = InterferometryResponse(observation, sky_domain, do_wgridding, epsilon, nthreads)
    w = observation.weight if weight is None else weight
    d = observation.vis if vis is None else vis
    vol = sky_domain[-1].scalar_dvol
    if weighting == "natural":
        return R.adjoint(d * w/w.s_sum()) / vol**2
    elif weighting == "uniform":
        w = uniform_weights(observation, sky_domain)
        return R.adjoint(d * w/w.s_sum()) / vol**2
    raise RuntimeError


def uvw_density(eff_u, eff_v, sky_domain, weights):
    """

    """
    if weights is not None:
        assert weights.shape == eff_u.shape
        weights = weights.ravel()
    u, v = eff_u.ravel(), eff_v.ravel()
    _, _, _, sdom = sky_domain
    dstx, dsty = sdom.distances
    nx, ny = sdom.shape
    ku = np.sort(np.fft.fftfreq(nx, dstx))
    kv = np.sort(np.fft.fftfreq(ny, dsty))
    assert np.min(u) >= ku[0]
    assert np.max(u)  < ku[-1]
    assert np.min(v) >= kv[0]
    assert np.max(v)  < kv[-1]
    H, xedges, yedges = np.histogram2d(u, v, bins=[ku, kv], weights=weights)
    # import matplotlib.pyplot as plt
    # plt.imshow(H)
    # plt.show()
    # plt.close()
    return H, xedges, yedges


def uniform_weights(observation, sky_domain):
    """

    """
    weights = np.empty_like(observation.weight.val)
    u, v = observation.effective_uvw()[0:2]
    for ipol in range(observation.npol):
        Hnorm, xedges0, yedges0 = uvw_density(u, v, sky_domain, None)
        H    , xedges , yedges  = uvw_density(u, v, sky_domain, observation.weight.val[ipol])
        assert np.all(xedges == xedges0)
        assert np.all(yedges == yedges0)
        xindices = np.searchsorted(xedges, u.ravel())
        yindices = np.searchsorted(yedges, v.ravel())
        norm = Hnorm[xindices-1, yindices-1].reshape(weights.shape[1:])
        norm *= norm  # FIXME Why
        weights[ipol] = H[xindices-1, yindices-1].reshape(weights.shape[1:]) / norm
    return ift.makeField(observation.weight.domain, weights)
