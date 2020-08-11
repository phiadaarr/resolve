# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift

from .observation import Observation
from .response import StokesIResponse
from .util import my_assert, my_asserteq


class ImagingLikelihood(ift.Operator):
    def __init__(self, observation, sky_operator):
        my_assert(isinstance(observation, Observation))
        my_assert(isinstance(sky_operator, ift.Operator))
        R = StokesIResponse(observation, sky_operator.target)
        my_asserteq(R.target.shape, observation.vis.shape)
        invcov = ift.makeOp(ift.makeField(R.target, observation.weight))
        vis = ift.makeField(R.target, observation.vis)
        op = ift.GaussianEnergy(mean=vis, inverse_covariance=invcov) @ R @ sky_operator
        self._domain, self._target = op.domain, op.target
        self.apply = op.apply


class ImagingCalibrationLikelihood(ift.Operator):
    def __init__(self, observation, sky_operator, calibration_operator):
        raise NotImplementedError


class CalibrationLikelihood(ift.Operator):
    def __init__(self, observation, calibration_operator, model_visibilities):
        raise NotImplementedError
