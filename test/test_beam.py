# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift

import resolve as rve
from .common import setup_function, teardown_function


def test_alma_beam():
    npix = 50
    dst = rve.str2rad("3as")
    sdom = ift.RGSpace([npix, npix], [dst, dst])
    fdom = rve.IRGSpace([1e9])
    dom = rve.default_sky_domain(sdom=sdom, fdom=fdom)
    beam = rve.alma_beam(dom, 10.5, 0.75)
