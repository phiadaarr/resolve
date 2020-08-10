# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift

from .observation import Observation
from .response import StokesIResponse
from .util import my_assert, my_asserteq, complex2float_dtype


class ImagingLikelihood(ift.Operator):
    def __init__(self, observation, sky_operator, nthreads, epsilon):
        my_assert(isinstance(observation, Observation))
        my_assert(isinstance(sky_operator, ift.Operator))

        # FIXME Will be removed as soon as ducc can figure that out itself
        wstacking = False
        R = StokesIResponse(observation, sky_operator.target, nthreads, epsilon, wstacking)
        # TEMP
        ift.extra.check_linear_operator(R, target_dtype=observation.vis.dtype,
                                        domain_dtype=complex2float_dtype(observation.vis.dtype),
                                        only_r_linear=True, rtol=epsilon, atol=epsilon)
        my_asserteq(R.target.shape, observation.vis.shape)


class ImagingCalibrationLikelihood(ift.Operator):
    def __init__(self, observation, sky_operator, calibration_operator):
        raise NotImplementedError


class CalibrationLikelihood(ift.Operator):
    def __init__(self, observation, calibration_operator, model_visibilities):
        raise NotImplementedError
