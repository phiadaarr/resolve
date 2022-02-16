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

import numpy as np
import nifty8 as ift

from . import Pybind11Operator


def PolarizationMatrixExponential(target, nthreads=1):
    from . import _cpp
    pdom = target[0]
    dom = {kk: target[1:] for kk in pdom.labels}
    return Pybind11Operator(dom, target, _cpp.PolarizationMatrixExponential(nthreads))
