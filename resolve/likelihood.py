# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import numpy as np

import nifty8 as ift

from .data.observation import Observation
from .response import FullPolResponse, MfResponse, StokesIResponse
from .util import my_assert, my_assert_isinstance, my_asserteq


def _get_mask(observation):
    # Only needed for variable covariance gaussian energy
    my_assert_isinstance(observation, Observation)
    vis = observation.vis
    flags = observation.flags
    if not np.any(flags.val):
        return ift.ScalingOperator(vis.domain, 1.0), vis, observation.weight
    mask = observation.mask_operator
    return mask, mask(vis), mask(observation.weight)


def get_mask_multi_field(weight):
    assert isinstance(weight, ift.MultiField)
    op = []
    for kk, ww in weight.items():
        flags = ww.val == 0.0
        if np.any(flags):
            myop = ift.MaskOperator(ift.makeField(ww.domain, flags))
        else:
            myop = ift.ScalingOperator(ww.domain, 1.0)
        op.append(myop.ducktape(kk).ducktape_left(kk))
    op = reduce(add, op)
    assert op.domain == weight.domain
    return op


def _Likelihood(operator, data, metric_at_pos, model_data):
    my_assert_isinstance(operator, model_data, ift.Operator)
    my_asserteq(operator.target, ift.DomainTuple.scalar_domain())
    my_asserteq(model_data.target, data.domain)
    operator.data = data
    operator.metric_at_pos = metric_at_pos
    operator.model_data = model_data
    return operator


def _build_gauss_lh_nres(op, mean, invcov):
    my_assert_isinstance(op, ift.Operator)
    my_assert_isinstance(mean, invcov, (ift.Field, ift.MultiField))
    my_asserteq(op.target, mean.domain, invcov.domain)

    lh = ift.GaussianEnergy(mean=mean, inverse_covariance=ift.makeOp(invcov, sampling_dtype=mean.dtype)) @ op
    return _Likelihood(lh, mean, lambda x: ift.makeOp(invcov), op)


def _varcov(observation, Rs, inverse_covariance_operator):
    from .simple_operators import KeyPrefixer
    mosaicing = isinstance(observation, dict)
    s0, s1 = "residual", "inverse covariance"
    if mosaicing:
        lhs = []
        vis = {}
        masks = []
        for kk, oo in observation.items():
            mask = oo.mask_operator
            masks.append(mask.ducktape(kk).ducktape_left(kk))
            tgt = mask.target
            vis[kk] = mask(oo.vis)
            dtype = oo.vis.dtype
            a = ift.Adder(vis[kk], neg=True).ducktape_left(s0).ducktape("modeld" + kk)
            b = ift.ScalingOperator(mask.target, 1.).ducktape_left(s1).ducktape("icov"+kk)
            e = ift.VariableCovarianceGaussianEnergy(tgt, s0, s1, dtype)
            lhs.append(e @ (a+b))
        masks = reduce(add, masks)

        a = KeyPrefixer(masks.target, "modeld") @ masks @ Rs
        b = KeyPrefixer(masks.target, "icov") @ masks @ inverse_covariance_operator
        lh = reduce(add, lhs) @ (a + b)

        vis = ift.MultiField.from_dict(vis)
        model_data = masks @ Rs
        icov_at = lambda x: ift.makeOp((masks @ inverse_covariance_operator).force(x))
    else:
        my_assert_isinstance(inverse_covariance_operator, ift.Operator)
        my_asserteq(Rs.target, observation.vis.domain, inverse_covariance_operator.target)
        mask, vis, _ = _get_mask(observation)
        residual = ift.Adder(vis, neg=True) @ mask @ Rs
        inverse_covariance_operator = mask @ inverse_covariance_operator
        dtype = observation.vis.dtype
        op = residual.ducktape_left(s0) + inverse_covariance_operator.ducktape_left(s1)
        lh = ift.VariableCovarianceGaussianEnergy(residual.target, s0, s1, dtype) @ op
        model_data = mask @ Rs
        icov_at = lambda x: ift.makeOp(inverse_covariance_operator.force(x))
    return _Likelihood(lh, vis, icov_at, model_data)


def ImagingLikelihood(
    observation,
    sky_operator,
    inverse_covariance_operator=None,
    calibration_operator=None,
):
    """Versatile likelihood class.

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
    observation : Observation or dict(Observation)
        Observation object from which observation.vis and potentially
        observation.weight is used for computing the likelihood.

    sky_operator : Operator
        Operator that generates sky. Needs to have as target:

        dom = (pdom, tdom, fdom, sdom)

        where `pdom` is a `PolarizationSpace`, `tdom` and `fdom` are an
        `IRGSpace`, and `sdom` is a two-dimensional `RGSpace`.

    inverse_covariance_operator : Operator
        Optional. Target needs to be the same space as observation.vis. If it is
        not specified, observation.wgt is taken as covariance.

    calibration_operator : Operator
        Optional. Target needs to be the same as observation.vis.

    """
    my_assert_isinstance(sky_operator, ift.Operator)
    model_data = InterferometryResponse(observation, sky_operator.target) @ sky_operator
    if inverse_covariance_operator is None:
        mask, vis, weight = _get_mask(observation)
        return _build_gauss_lh_nres(mask @ model_data, vis, weight)
    return _varcov(observation, model_data, inverse_covariance_operator)


def CalibrationLikelihood(
    observation,
    calibration_operator,
    model_visibilities,
    inverse_covariance_operator=None,
):
    """Versatile calibration likelihood class

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
        mask, vis, wgt = _get_mask(observation)
        return _build_gauss_lh_nres(mask @ model_data, vis, wgt)
    my_assert_isinstance(inverse_covariance_operator, ift.Operator)
    return _varcov(observation, model_data, inverse_covariance_operator)
