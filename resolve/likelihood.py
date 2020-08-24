# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

from .observation import Observation
from .response import StokesIResponse
from .util import my_assert_isinstance, my_asserteq

# TODO VariableCovariance version for all likelihoods


def _get_mask(observation):
    my_assert_isinstance(observation, Observation)
    vis = observation.vis
    if np.all(observation.flags.val):
        return ift.ScalingOperator(vis.domain, 1.), vis, observation.weight
    mask = ift.MaskOperator(ift.makeField(vis.domain, ~observation.flags.val.astype(bool)))
    return mask, mask(vis), mask(observation.weight)


def ImagingLikelihood(observation, sky_operator):
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, ift.Operator)
    R = StokesIResponse(observation, sky_operator.target)
    my_asserteq(R.target, observation.vis.domain)
    mask, vis, invcov = _get_mask(observation)
    return ift.GaussianEnergy(mean=vis, inverse_covariance=ift.makeOp(invcov)) @ mask @ R @ sky_operator


def ImagingLikelihoodVariableCovariance(observation, sky_operator, inverse_covariance_operator):
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, inverse_covariance_operator, ift.Operator)
    my_assert_isinstance(inverse_covariance_operator.domain, sky_operator.domain, ift.MultiDomain)
    my_assert_isinstance(inverse_covariance_operator.target, sky_operator.target, ift.DomainTuple)
    R = StokesIResponse(observation, sky_operator.target)
    my_asserteq(R.target.shape, observation.vis.shape)
    mask, vis, _ = _get_mask(observation)
    residual = ift.Adder(vis, neg=True) @ mask @ R @ sky_operator
    inverse_covariance_operator = mask @ inverse_covariance_operator
    op = residual.ducktape_left('r') + inverse_covariance_operator.ducktape_left('ic')
    return ift.VariableCovarianceGaussianEnergy(residual.target, 'r', 'ic', observation.vis.dtype) @ op


def ImagingCalibrationLikelihood(observation, sky_operator, calibration_operator):
    if observation.npol == 1:
        print('Warning: Use calibration with only one polarization present.')
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, calibration_operator, ift.Operator)
    my_assert_isinstance(calibration_operator.domain, ift.MultiDomain)
    R = StokesIResponse(observation, sky_operator.target)
    my_asserteq(R.target, observation.vis.domain, calibration_operator.target)
    mask, vis, invcov = _get_mask(observation)
    modelvis = mask @ (calibration_operator*(R @ sky_operator))
    return ift.GaussianEnergy(mean=vis, inverse_covariance=ift.makeOp(invcov)) @ modelvis


def CalibrationLikelihood(observation, calibration_operator, model_visibilities):
    if observation.npol == 1:
        print('Warning: Use calibration with only one polarization present.')
    my_assert_isinstance(calibration_operator.domain, ift.MultiDomain)
    my_asserteq(calibration_operator.target, model_visibilities.domain, observation.vis.domain)
    mask, vis, invcov = _get_mask(observation)
    return ift.GaussianEnergy(vis, ift.makeOp(invcov)) @ ift.makeOp(mask(model_visibilities)) @ mask @ calibration_operator
