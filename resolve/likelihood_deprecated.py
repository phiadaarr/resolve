# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add
from warnings import warn

import nifty8 as ift
import numpy as np

from .data.observation import Observation
from .response_new import InterferometryResponse
from .util import (_duplicate, _obj2list, my_assert, my_assert_isinstance,
                   my_asserteq)
from .energy_operators import DiagonalGaussianLikelihood
from .likelihood import _get_mask
from .dtype_converter import DtypeConverter


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


def _build_gauss_lh_nres(op, mean, invcov):
    my_assert_isinstance(op, ift.Operator)
    my_assert_isinstance(mean, invcov, (ift.Field, ift.MultiField))
    my_asserteq(op.target, mean.domain, invcov.domain)
    dt = DtypeConverter(op.target, np.complex128, mean.dtype)
    return DiagonalGaussianLikelihood(data=mean, inverse_covariance=invcov) @ dt @ op


def _varcov(observation, Rs, inverse_covariance_operator):
    s0, s1 = "residual", "inverse covariance"
    my_assert_isinstance(inverse_covariance_operator, ift.Operator)
    my_asserteq(Rs.target, observation.vis.domain, inverse_covariance_operator.target)
    mask, vis, _ = _get_mask(observation)
    residual = ift.Adder(vis, neg=True) @ mask @ Rs
    inverse_covariance_operator = mask @ inverse_covariance_operator
    dtype = observation.vis.dtype
    op = residual.ducktape_left(s0) + inverse_covariance_operator.ducktape_left(s1)
    return ift.VariableCovarianceGaussianEnergy(residual.target, s0, s1, dtype) @ op


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
    warn("`calibration_operator` won't be part of the next release. "
         "Switch to `ImagingLikelihood`.", category=DeprecationWarning)
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
