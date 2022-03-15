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
import numpy as np

from ..data.observation import SingleDishObservation


def SingleDishResponse(
    observation, domain, beam_function, global_phase_center, additive_term=None
):
    assert isinstance(observation, SingleDishObservation)
    domain = ift.makeDomain(domain)
    assert len(domain) == 1
    codomain = domain[0].get_default_codomain()
    kernel = codomain.get_conv_kernel_from_func(beam_function)
    HT = ift.HartleyOperator(domain, codomain)
    conv = HT.inverse @ ift.makeOp(kernel) @ HT.scale(domain.total_volume())
    # FIXME Move into tests
    fld = ift.from_random(conv.domain)
    ift.extra.assert_allclose(conv(fld).integrate(), fld.integrate())

    pc = observation.pointings.phase_centers.T - np.array(global_phase_center)[:, None]
    pc = pc + (np.array(domain.shape) * np.array(domain[0].distances) / 2)[:, None]
    # Convention: pointing convention (see also BeamDirection)
    pc[0] *= -1
    interp = ift.LinearInterpolator(domain, pc)
    bc = ift.ContractionOperator(observation.vis.domain, (0, 2)).adjoint
    # NOTE The volume factor above `domain.total_volume()` and the volume factor
    # below `domain[0].scalar_dvol` cancel each other. They are left in the
    # code such that the convolution leaves the integral invariant.

    convsky = conv.scale(domain[0].scalar_dvol).ducktape("sky")
    if additive_term is not None:
        convsky = convsky + additive_term
    return bc @ interp @ convsky
