# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2020-2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

from functools import reduce
from operator import add
from warnings import warn

import nifty8 as ift
import numpy as np

from .data.observation import Observation
from .dtype_converter import DtypeConverter
from .energy_operators import (
    DiagonalGaussianLikelihood,
    VariableCovarianceDiagonalGaussianLikelihood,
)
from .response import InterferometryResponse
from .util import _duplicate, _obj2list, my_assert, my_assert_isinstance, my_asserteq


def ImagingLikelihood(
    observation,
    sky_operator,
    epsilon,
    do_wgridding,
    log_inverse_covariance_operator=None,
    calibration_operator=None,
    calibration_field=None,
    verbosity=0,
    nthreads=1,
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

    epsilon : float

    do_wgridding : bool

    log_inverse_covariance_operator : Operator or list of Operator
        Optional. Target needs to be the same space as observation.vis. If it
        is not specified, observation.wgt is taken as covariance.

    calibration_operator : Operator or list of Operator
        Optional. Target needs to be the same as observation.vis.

    calibration_field: Field or list of Field
        Optional. Domain needs to be the same as observation.vis.

    verbosity : int

    nthreads : int

    Note
    ----
    For each observation only either calibration_operator or calibration_field
    can be set.
    """
    my_assert_isinstance(sky_operator, ift.Operator)
    obs = _obj2list(observation, Observation)
    cops = _duplicate(_obj2list(calibration_operator, ift.Operator), len(obs))
    cflds = _duplicate(_obj2list(calibration_field, ift.Field), len(obs))
    log_icovs = _duplicate(_obj2list(log_inverse_covariance_operator, ift.Operator), len(obs))
    if len(obs) == 0:
        raise ValueError("List of observations is empty")

    internal_sky_key = "_sky"

    energy = []
    for ii, (oo, cop, cfld, log_icov) in enumerate(zip(obs, cops, cflds, log_icovs)):
        dtype = oo.vis.dtype

        if cfld is not None and cop is not None:
            raise ValueError(
                "Setting a calibration field and a calibration operator at the "
                "same time, does not work."
            )

        R = InterferometryResponse(
            oo,
            sky_operator.target,
            do_wgridding=do_wgridding,
            epsilon=epsilon,
            verbosity=verbosity,
            nthreads=nthreads,
        ).ducktape(internal_sky_key)
        if cop is not None:
            from .dtype_converter import DtypeConverter

            dt = DtypeConverter(cop.target, np.complex128, dtype)
            R = (dt @ cop) * R  # Apply calibration solutions
        if cfld is not None:
            if cfld.dtype != dtype:
                raise ValueError(
                    f"Calibration solution field ({cfld.dtype}) needs "
                    "to have the same dtype as the observation ({dtype})."
                )
            R = ift.makeOp(cfld) @ R  # Apply calibration solutions

        if log_icov is None:
            ee = (
                DiagonalGaussianLikelihood(
                    data=oo.vis, inverse_covariance=oo.weight, mask=oo.mask
                )
                @ R
            )
            ee.name = f"{oo.source_name} (data wgts) {ii}"
        else:
            s0, s1 = "_model_data", "_log_icov"
            ee = VariableCovarianceDiagonalGaussianLikelihood(
                data=oo.vis, key_signal=s0, key_log_inverse_covariance=s1, mask=oo.mask
            ) @ (log_icov.ducktape_left(s1) + R.ducktape_left(s0))
            ee.name = f"{oo.source_name} (varcov) {ii}"
        energy.append(ee)
    energy = reduce(add, energy)
    sky_operator = sky_operator.ducktape_left(internal_sky_key)
    return energy.partial_insert(sky_operator)


def CalibrationLikelihood(
    observation,
    calibration_operator,
    model_visibilities,
    log_inverse_covariance_operator=None,
    nthreads=1,
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

    log_inverse_covariance_operator : Operator or list of Operators
        Optional. Target needs to be the same space as observation.vis. If it is
        not specified, observation.wgt is taken as covariance.

    nthreads : int
    """
    obs = _obj2list(observation, Observation)
    cops = _duplicate(_obj2list(calibration_operator, ift.Operator), len(obs))
    log_icovs = _duplicate(_obj2list(log_inverse_covariance_operator, ift.Operator), len(obs))
    model_d = _duplicate(_obj2list(model_visibilities, ift.Field), len(obs))
    model_d = [ift.makeOp(mm) @ cop for mm, cop in zip(model_d, cops)]

    if len(obs) > 1:
        raise NotImplementedError
    obs, model_d, log_icov = obs[0], model_d[0], log_icovs[0]

    dt = DtypeConverter(model_d.target, np.complex128, obs.vis.dtype)
    dt_icov = DtypeConverter(model_d.target, np.float64, obs.weight.dtype)

    if log_icov is None:
        e = DiagonalGaussianLikelihood(
            data=obs.vis,
            inverse_covariance=obs.weight,
            nthreads=nthreads,
            mask=obs.mask,
        )
        return e @ dt @ model_d
    else:
        s0, s1 = "model data", "inverse covariance"
        e = VariableCovarianceDiagonalGaussianLikelihood(
            obs.vis, s0, s1, mask=obs.mask, nthreads=nthreads
        )
        return e @ ((dt @ model_d).ducktape_left(s0) + (dt_icov @ log_icov).ducktape_left(s1))
