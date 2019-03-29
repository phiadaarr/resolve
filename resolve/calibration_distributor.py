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
# Copyright(C) 2019 Max-Planck-Society

import numpy as np
import nifty5 as ift


def CalibrationDistributor(domain, ant, time):
    assert len(time.shape) == 1
    assert ant.shape == time.shape
    assert time.min() >= 0
    assert time.max() < domain[1].total_volume

    dd = [ift.RGSpace(domain[0].shape, distances=1.), domain[1]]
    dd = ift.DomainTuple.make(dd)
    positions = np.array([ant, time])
    li = ift.LinearInterpolator(dd, positions)
    return li @ ift.Realizer(li.domain) @ ift.GeometryRemover(dd, 0).adjoint
