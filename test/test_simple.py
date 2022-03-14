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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import nifty8 as ift
import pytest

import resolve as rve

from .common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize

pdom = list2fixture([rve.PolarizationSpace("I"),
                     rve.PolarizationSpace(["I", "Q", "U"]),
                     rve.PolarizationSpace(["I", "Q", "U", "V"]),
                     ])

fdom = list2fixture([rve.IRGSpace([12., 13.])])
sdom = list2fixture([ift.RGSpace([12., 13.])])


def test_multi_field_stacker(pdom, fdom, sdom):
    domain = rve.default_sky_domain(pdom=pdom, fdom=fdom, sdom=sdom)
    op = rve.MultiFieldStacker(domain, 0, domain[0].labels)
    ift.extra.check_linear_operator(op)
