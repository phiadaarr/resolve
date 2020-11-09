# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

from .observation import Observation
from .response import StokesIResponse, MfResponse
from .util import my_assert_isinstance, my_asserteq, my_assert

# FIXME VariableCovariance version for all likelihoods


def _get_mask(observation):
    # Only needed for variable covariance gaussian energy
    my_assert_isinstance(observation, Observation)
    vis = observation.vis
    flags = observation.flags
    if not np.any(flags):
        return ift.ScalingOperator(vis.domain, 1.), vis, observation.weight
    mask = ift.MaskOperator(ift.makeField(vis.domain, flags))
    return mask, mask(vis), mask(observation.weight)


class _Likelihood(ift.EnergyOperator):
    def __init__(self, operator, normalized_residual_operator):
        my_assert_isinstance(operator, ift.Operator)
        my_asserteq(operator.target, ift.DomainTuple.scalar_domain())
        my_assert_isinstance(normalized_residual_operator, ift.Operator)
        self._op = operator
        self._domain = operator.domain
        self.apply = operator.apply
        self._nres = normalized_residual_operator

    @property
    def normalized_residual(self):
        return self._nres

    def __repr__(self):
        return self._op.__repr__()


def _build_gauss_lh_nres(op, mean, invcov):
    my_assert_isinstance(op, ift.Operator)
    my_assert_isinstance(mean, invcov, (ift.Field, ift.MultiField))
    my_asserteq(op.target, mean.domain, invcov.domain)
    lh = ift.GaussianEnergy(mean=mean, inverse_covariance=ift.makeOp(invcov)) @ op
    nres = ift.makeOp(invcov.sqrt()) @ ift.Adder(mean, neg=True) @ op
    return _Likelihood(lh, nres)


def _build_varcov_gauss_lh_nres(residual, inverse_covariance, dtype):
    my_assert_isinstance(residual, inverse_covariance, ift.Operator)
    my_asserteq(residual.target, inverse_covariance.target)
    op = residual.ducktape_left('r') + inverse_covariance.ducktape_left('ic')
    lh = ift.VariableCovarianceGaussianEnergy(residual.target, 'r', 'ic', dtype) @ op
    nres = residual*inverse_covariance.sqrt()
    return _Likelihood(lh, nres)


def ImagingLikelihood(observation, sky_operator):
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, ift.Operator)
    R = StokesIResponse(observation, sky_operator.target)
    my_asserteq(R.target, observation.vis.domain)
    return _build_gauss_lh_nres(R @ sky_operator, observation.vis, observation.weight)


def MfImagingLikelihood(observation, sky_operator):
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, ift.Operator)
    R = MfResponse(observation, sky_operator.target)
    # FIXME Move to tests
    # ift.extra.check_linear_operator(R, rtol=1e-5, target_dtype=np.complex128, only_r_linear=True)
    return _build_gauss_lh_nres(R @ sky_operator, observation.vis, observation.weight)


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
    dtype = observation.vis.dtype
    return _build_varcov_gauss_lh_nres(residual, inverse_covariance_operator, dtype)


def ImagingCalibrationLikelihood(observation, sky_operator, calibration_operator):
    if observation.npol == 1:
        print('Warning: Use calibration with only one polarization present.')
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, calibration_operator, ift.Operator)
    my_assert_isinstance(calibration_operator.domain, ift.MultiDomain)
    R = StokesIResponse(observation, sky_operator.target)
    my_asserteq(R.target, observation.vis.domain, calibration_operator.target)
    modelvis = calibration_operator*(R @ sky_operator)
    return _build_gauss_lh_nres(modelvis, observation.vis, observation.weight)


def CalibrationLikelihood(observation, calibration_operator, model_visibilities):
    if observation.npol == 1:
        print('Warning: Use calibration with only one polarization present.')
    my_assert_isinstance(calibration_operator.domain, ift.MultiDomain)
    my_asserteq(calibration_operator.target, model_visibilities.domain, observation.vis.domain)
    return _build_gauss_lh_nres(ift.makeOp(model_visibilities) @ calibration_operator,
                                observation.vis, observation.weight)
