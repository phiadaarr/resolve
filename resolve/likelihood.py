# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np

from .data.observation import Observation
from .response_new import InterferometryResponse
from .util import _duplicate, _obj2list, my_assert_isinstance
from .energy_operators import DiagonalGaussianLikelihood, VariableCovarianceDiagonalGaussianLikelihood


def _get_mask(observation):
    # Only needed for variable covariance gaussian energy
    my_assert_isinstance(observation, Observation)
    vis = observation.vis
    flags = observation.flags
    if not np.any(flags.val):
        return ift.ScalingOperator(vis.domain, 1.0), vis, observation.weight
    mask = observation.mask_operator
    return mask, mask(vis), mask(observation.weight)


def ImagingLikelihood(
    observation,
    sky_operator,
    log_inverse_covariance_operator=None,
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

    log_inverse_covariance_operator : Operator or list of Operator
        Optional. Target needs to be the same space as observation.vis. If it
        is not specified, observation.wgt is taken as covariance.

    calibration_operator : Operator or list of Operator
        Optional. Target needs to be the same as observation.vis.

    """
    my_assert_isinstance(sky_operator, ift.Operator)
    obs = _obj2list(observation, Observation)
    cops = _duplicate(_obj2list(calibration_operator, ift.Operator), len(obs))
    log_icovs = _duplicate(
        _obj2list(log_inverse_covariance_operator, ift.Operator), len(obs)
    )
    if len(obs) == 0:
        raise ValueError("List of observations is empty")

    internal_sky_key = "_sky"

    energy = []
    for ii, (oo, cop, log_icov) in enumerate(zip(obs, cops, log_icovs)):
        mask, vis, weight = _get_mask(oo)
        if log_icov is not None:
            log_icov = mask @ log_icov
        dtype = oo.vis.dtype

        R = InterferometryResponse(oo, sky_operator.target).ducktape(internal_sky_key)
        if cop is not None:
            from .dtype_converter import DtypeConverter
            dt = DtypeConverter(cop.target, np.complex128, dtype)
            R = (dt @ cop) * R  # Apply calibration solutions
        R = mask @ R  # Apply flags  FIXME Move this into cpp likelihoods

        if log_icov is None:
            ee = DiagonalGaussianLikelihood(data=vis, inverse_covariance=weight) @ R
            ee.name = f"{oo.source_name} (data wgts)"
        else:
            s0, s1 = "_model_data", "_log_icov"
            ee = VariableCovarianceDiagonalGaussianLikelihood(
                data=vis, key_signal=s0, key_log_inverse_covariance=s1
            ) @ (log_icov.ducktape_left(s1) + R.ducktape_left(s0))
            ee.name = f"{oo.source_name} (varcov)"
        energy.append(ee)
    energy = reduce(add, energy)
    sky_operator = sky_operator.ducktape_left(internal_sky_key)
    return energy.partial_insert(sky_operator)
