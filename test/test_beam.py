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
