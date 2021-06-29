# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

from ..observation import SingleDishObservation


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
