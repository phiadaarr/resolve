# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2022 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np

from .constants import str2rad
from .irg_space import IRGSpace
from .polarization_space import PolarizationSpace
from .response_new import InterferometryResponse
from .global_config import set_epsilon, set_wgridding


def dirty_image(observation, weighting, fov, npix, freqs=[1.0], times=[0.0]):
    if isinstance(fov, (str, float)):
        fov = [fov, fov]
    fov = np.array(list(map(str2rad, fov)))
    if isinstance(npix, int):
        npix = [npix, npix]
    npix = np.array(npix)
    sdom = ift.RGSpace(npix, fov / npix)

    pol = observation.polarization.has_crosshanded()
    if pol:
        pdom = PolarizationSpace(["I", "Q", "U", "V"])
    else:
        pdom = PolarizationSpace("I")

    fdom = IRGSpace(freqs)
    tdom = IRGSpace(times)
    R = InterferometryResponse(observation, (pdom, tdom, fdom, sdom))

    if weighting == "natural":
        w = observation.weight
    elif weighting == "uniform":
        # FIXME Figure out units
        w = observation.weight
        d = observation.vis
        rho = (R @ R.adjoint)(ift.full(R.target, 1.+0.j))
        return R.adjoint(d/rho*w)
    else:
        raise RuntimeError

    return R.adjoint(w * observation.vis)
