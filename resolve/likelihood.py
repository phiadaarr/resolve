# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np

from .data.observation import Observation
from .response_new import InterferometryResponse
from .util import my_assert, my_assert_isinstance, my_asserteq, _obj2list, _duplicate


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
    s0, s1 = "residual", "inverse covariance"
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
    observation : Observation or list of Observation
        Observation objects from which vis, uvw, freq and potentially weight
        are used for computing the likelihood.

    sky_operator : Operator
        Operator that generates sky. Needs to have as target:

        dom = (pdom, tdom, fdom, sdom)

        where `pdom` is a `PolarizationSpace`, `tdom` and `fdom` are an
        `IRGSpace`, and `sdom` is a two-dimensional `RGSpace`.

    inverse_covariance_operator : Operator or list of Operator
        Optional. Target needs to be the same space as observation.vis. If it
        is not specified, observation.wgt is taken as covariance.

    calibration_operator : Operator or list of Operator
        Optional. Target needs to be the same as observation.vis.

    """
    my_assert_isinstance(sky_operator, ift.Operator)
    obs = _obj2list(observation, Observation)
    cops = _duplicate(_obj2list(calibration_operator, ift.Operator), len(obs))
    icovs = _duplicate(_obj2list(inverse_covariance_operator, ift.Operator),
                       len(obs))

    energy = []
    data, model_data, icov_at = [], [], []
    used_keys = []
    for ii, (oo, cop, icov) in enumerate(zip(obs, cops, icovs)):
        virtual_key = f"_{ii} {oo.source_name}"
        assert virtual_key not in used_keys
        mask, vis, weight = _get_mask(oo)
        dtype = oo.vis.dtype

        R = InterferometryResponse(oo, sky_operator.target).ducktape("_sky")
        if cop is not None:
            R = cop*R  # Apply calibration solutions
        R = mask @ R  # Apply flags

        if icov is None:
            icov = ift.makeOp(weight, sampling_dtype=dtype)
            ee = ift.GaussianEnergy(mean=vis, inverse_covariance=icov) @ R
            icov_at.append(lambda x: ift.BlockDiagonalOperator(ift.makeDomain({virtual_key: icov.domain}), {virtual_key: icov}))
        else:
            s0, s1 = "_resi", "_icov"
            resi = ift.Adder(vis, neg=True) @ R
            icov = mask @ icov
            op = resi.ducktape_left(s0) + icov.ducktape_left(s1)
            ee = ift.VariableCovarianceGaussianEnergy(resi.target, s0, s1, dtype) @ \
                    (resi.ducktape_left(s0) + icov.ducktape_left(s1))
            icov_at.append(lambda x: ift.makeOp(icov.ducktape_left(virtual_key).force(x)))

        energy.append(ee)
        data.append(vis.ducktape_left(virtual_key))
        model_data.append(R.ducktape_left(virtual_key))

        used_keys.append(virtual_key)

    sky_operator = sky_operator.ducktape_left("_sky")
    energy = reduce(add, energy).partial_insert(sky_operator)
    data = ift.MultiField.union(data)
    model_data = reduce(add, model_data).partial_insert(sky_operator)
    total_icov_at = lambda x: reduce(add, (iicc(x) for iicc in icov_at))
    return _Likelihood(energy, data, total_icov_at, model_data)


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
    observation : Observation or list of Observations
        Observation object from which observation.vis and potentially
        observation.weight is used for computing the likelihood.

    calibration_operator : Operator or list of Operators
        Target needs to be the same as observation.vis.

    model_visibilities : Field or list of Fields
        Known model visiblities that are used for calibration. Needs to be
        defined on the same domain as `observation.vis`.

    inverse_covariance_operator : Operator or list of Operators
        Optional. Target needs to be the same space as observation.vis. If it is
        not specified, observation.wgt is taken as covariance.
    """
    obs = _obj2list(observation, Observation)
    cops = _duplicate(_obj2list(calibration_operator, ift.Operator), len(obs))
    icovs = _duplicate(_obj2list(inverse_covariance_operator, ift.Operator),
                       len(obs))
    model_d = _duplicate(_obj2list(model_visibilities, ift.Field), len(obs))
    model_d = [ift.makeOp(mm) @ cop for mm, cop in zip(model_d, cops)]

    if len(obs) > 1:
        raise NotImplementedError
    if icovs[0] is None:
        mask, vis, wgt = _get_mask(obs[0])
        return _build_gauss_lh_nres(mask @ model_d[0], vis, wgt)
    return _varcov(obs[0], model_d[0], icov[0])
