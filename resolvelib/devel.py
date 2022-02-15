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
# Copyright(C) 2021-2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import resolve as rve
import nifty8 as ift
import resolvelib

from cpp2py import Pybind11Operator


pdom = rve.PolarizationSpace(["I", "Q", "U"])
sdom = ift.RGSpace([4, 4])

dom = rve.default_sky_domain(pdom=pdom, sdom=sdom)
dom = {kk: dom[1:] for kk in pdom.labels}

op = Pybind11Operator(dom, dom, resolvelib.PolarizationMatrixExponential())

loc = ift.from_random(op.domain)
op(loc)
exit()

ift.extra.check_operator(op, loc)
