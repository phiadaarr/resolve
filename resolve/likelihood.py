# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

from .observation import Observation
from .response import FullPolResponse, MfResponse, StokesIResponse
from .util import my_assert_isinstance, my_asserteq


def _get_mask(observation):
    # Only needed for variable covariance gaussian energy
    my_assert_isinstance(observation, Observation)

    vis = observation.vis
    flags = observation.flags
    if not np.any(flags):
        return ift.ScalingOperator(vis.domain, 1.0), vis, observation.weight
    mask = ift.MaskOperator(ift.makeField(vis.domain, flags))
    return mask, mask(vis), mask(observation.weight)


def _Likelihood(operator, normalized_residual_operator):
    my_assert_isinstance(operator, ift.Operator)
    my_asserteq(operator.target, ift.DomainTuple.scalar_domain())
    my_assert_isinstance(normalized_residual_operator, ift.Operator)

    operator.normalized_residual = normalized_residual_operator
    return operator


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

    s0, s1 = "residual", "inverse covariance"
    op = residual.ducktape_left(s0) + inverse_covariance.ducktape_left(s1)
    lh = ift.VariableCovarianceGaussianEnergy(residual.target, s0, s1, dtype) @ op
    nres = residual * inverse_covariance.sqrt()
    return _Likelihood(lh, nres)


def _varcov(observation, Rs, inverse_covariance_operator):
    my_assert_isinstance(inverse_covariance_operator, ift.Operator)
    my_asserteq(Rs.target, observation.vis.domain, inverse_covariance_operator.target)
    mask, vis, _ = _get_mask(observation)
    residual = ift.Adder(vis, neg=True) @ mask @ Rs
    inverse_covariance_operator = mask @ inverse_covariance_operator
    dtype = observation.vis.dtype
    return _build_varcov_gauss_lh_nres(residual, inverse_covariance_operator, dtype)


def ImagingLikelihood(
    observation,
    sky_operator,
    inverse_covariance_operator=None,
    calibration_operator=None,
):
    """Versatile likelihood class that automatically chooses the correct
    response class.

    Supports polarization imaging and Stokes I imaging. Supports single-frequency
    and multi-frequency imaging.

    If a calibration operator is passed, it returns an operator that computes:

    residual = calibration_operator * (R @ sky_operator)
    likelihood = 0.5 * residual^dagger @ inverse_covariance @ residual

    Otherwise, it returns an operator that computes:

    residual = R @ sky_operator
    likelihood = 0.5 * residual^dagger @ inverse_covariance @ residual

    If an inverse_covariance_operator is passed, it is inserted into the above
    formulae. If it is not passed, 1/observation.weights is used as inverse
    covariance.

    Parameters
    ----------
    observation : Observation
        Observation object from which observation.vis and potentially
        observation.weight is used for computing the likelihood.

    sky_operator : Operator
        Operator that generates sky. Can have as a target either a two-dimensional
        DomainTuple (single-frequency imaging), a three-dimensional DomainTuple
        (multi-frequency imaging) or a MultiDomain (polarization imaging).

    inverse_covariance_operator : Operator
        Optional. Target needs to be the same space as observation.vis. If it is
        not specified, observation.wgt is taken as covariance.

    calibration_operator : Operator
        Optional. Target needs to be the same as observation.vis.
    """
    my_assert_isinstance(observation, Observation)
    my_assert_isinstance(sky_operator, ift.Operator)
    sdom = sky_operator.target

    if isinstance(sdom, ift.MultiDomain):
        if len(sdom["I"].shape) == 3:
            raise NotImplementedError(
                "Polarization and multi-frequency at the same time not supported yet."
            )
        else:
            R = FullPolResponse(observation, sky_operator.target)
    else:
        if len(sdom.shape) == 3:
            R = MfResponse(observation, sdom)
        else:
            R = StokesIResponse(observation, sdom)
    model_data = R @ sky_operator

    if inverse_covariance_operator is None:
        return _build_gauss_lh_nres(model_data, observation.vis, observation.weight)
    return _varcov(observation, model_data, inverse_covariance_operator)


def CalibrationLikelihood(
    observation,
    calibration_operator,
    model_visibilities,
    inverse_covariance_operator=None,
):
    """Versatile calibration likelihood class that automatically chooses
    the correct response class.

    It returns an operator that computes:

    residual = calibration_operator * model_visibilities
    likelihood = 0.5 * residual^dagger @ inverse_covariance @ residual

    If an inverse_covariance_operator is passed, it is inserted into the above
    formulae. If it is not passed, 1/observation.weights is used as inverse
    covariance.

    Parameters
    ----------
    observation : Observation
        Observation object from which observation.vis and potentially
        observation.weight is used for computing the likelihood.

    calibration_operator : Operator
        Target needs to be the same as observation.vis.

    model_visibilities : Field or MultiField
        Known model visiblities that are used for calibration. Needs to be
        defined on the same domain as `observation.vis`.

    inverse_covariance_operator : Operator
        Optional. Target needs to be the same space as observation.vis. If it is
        not specified, observation.wgt is taken as covariance.
    """
    my_assert_isinstance(observation, Observation)
    my_assert(ift.is_fieldlike(model_visibilities))
    my_assert_isinstance(calibration_operator, ift.Operator)
    model_data = ift.makeOp(model_visibilities) @ calibration_operator
    if inverse_covariance_operator is None:
        return _build_gauss_lh_nres(model_data, observation.vis, observation.weight)
    my_assert_isinstance(inverse_covariance_operator, ift.Operator)
    return _varcov(observation, model_data, inverse_covariance_operator)
