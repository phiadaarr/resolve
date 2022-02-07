# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2022 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np

from .constants import str2rad
from .global_config import set_epsilon, set_wgridding
from .irg_space import IRGSpace
from .polarization_space import PolarizationSpace
from .response_new import InterferometryResponse
from .util import assert_sky_domain


def dirty_image(observation, weighting, sky_domain, vis=None, weight=None):
    assert_sky_domain(sky_domain)
    pol = observation.polarization.has_crosshanded()
    R = InterferometryResponse(observation, sky_domain)
    w = observation.weight if weight is None else weight
    d = observation.vis if vis is None else vis
    vol = sky_domain[-1].scalar_dvol
    if weighting == "natural":
        return R.adjoint(d * w/w.s_sum()) / vol**2
    elif weighting == "uniform":
        w1 = ift.full(R.target, 1.+0.j)
        rho = (R @ R.adjoint)(w1/w1.s_sum()) / vol**2
        return R.adjoint(d* 1/rho * w/w.s_sum()) / vol**2
    raise RuntimeError
