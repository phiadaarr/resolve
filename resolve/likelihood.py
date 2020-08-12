# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift

from .observation import Observation
from .response import StokesIResponse
from .util import my_assert_isinstance, my_asserteq


def ImagingLikelihood(observation, sky_operator):
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, ift.Operator)
    R = StokesIResponse(observation, sky_operator.target)
    my_asserteq(R.target, observation.vis.domain)
    invcov = ift.makeOp(observation.weight)
    return ift.GaussianEnergy(mean=observation.vis, inverse_covariance=invcov) @ R @ sky_operator


def ImagingLikelihoodVariableCovariance(observation, sky_operator, inverse_covariance_operator):
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, inverse_covariance_operator, ift.Operator)
    my_assert_isinstance(inverse_covariance_operator.domain, sky_operator.domain, ift.MultiDomain)
    my_assert_isinstance(inverse_covariance_operator.target, sky_operator.target, ift.DomainTuple)
    R = StokesIResponse(observation, sky_operator.target)
    my_asserteq(R.target.shape, observation.vis.shape)
    residual = ift.Adder(observation.vis, neg=True) @ R @ sky_operator
    op = residual.ducktape_left('r') + inverse_covariance_operator.ducktape_left('ic')
    return ift.VariableCovarianceGaussianEnergy(observation.vis.domain, 'r', 'ic', observation.vis.dtype) @ op


class ImagingCalibrationLikelihood(ift.Operator):
    def __init__(self, observation, sky_operator, calibration_operator):
        if observation.vis.shape[0] == 1:
            print('Warning: Use calibration with only one polarization present.')
        raise NotImplementedError


def CalibrationLikelihood(observation, calibration_operator, model_visibilities):
    if observation.vis.shape[0] == 1:
        print('Warning: Use calibration with only one polarization present.')
    my_assert_isinstance(calibration_operator.domain, ift.MultiDomain)
    my_asserteq(calibration_operator.target, model_visibilities.domain, observation.vis.domain)
    invcov = ift.makeOp(observation.weight)
    return ift.GaussianEnergy(observation.vis, invcov) @ ift.makeOp(model_visibilities) @ calibration_operator
