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

import nifty8 as ift
import numpy as np

from .data.observation import Observation
from .response import InterferometryResponse
from .util import _duplicate, _obj2list, my_assert_isinstance
from .energy_operators import DiagonalGaussianLikelihood, VariableCovarianceDiagonalGaussianLikelihood
from functools import reduce
from operator import add
from warnings import warn

import nifty8 as ift
import numpy as np

from .data.observation import Observation
from .util import (_duplicate, _obj2list, my_assert, my_assert_isinstance,
                   my_asserteq)
from .energy_operators import DiagonalGaussianLikelihood
from .dtype_converter import DtypeConverter
from .energy_operators import VariableCovarianceDiagonalGaussianLikelihood


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
    epsilon,
    do_wgridding,
    log_inverse_covariance_operator=None,
    calibration_operator=None,
    verbosity=0,
    nthreads=1
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

    verbosity : int

    nthreads : int
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

        R = InterferometryResponse(oo, sky_operator.target, do_wgridding=do_wgridding, epsilon=epsilon, verbosity=verbosity, nthreads=nthreads).ducktape(internal_sky_key)
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


def CalibrationLikelihood(
    observation,
    calibration_operator,
    model_visibilities,
    inverse_covariance_operator=None,
    nthreads=1
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

    nthreads : int
    """
    obs = _obj2list(observation, Observation)
    cops = _duplicate(_obj2list(calibration_operator, ift.Operator), len(obs))
    icovs = _duplicate(_obj2list(inverse_covariance_operator, ift.Operator),
                       len(obs))
    model_d = _duplicate(_obj2list(model_visibilities, ift.Field), len(obs))
    model_d = [ift.makeOp(mm) @ cop for mm, cop in zip(model_d, cops)]

    if len(obs) > 1:
        raise NotImplementedError
    obs, model_d, icov = obs[0], model_d[0], icovs[0]

    dt = DtypeConverter(model_d.target, np.complex128, obs.vis.dtype)
    dt_icov = DtypeConverter(model_d.target, np.float64, obs.weight.dtype)

    mask, vis, wgt = _get_mask(obs)
    model_d = dt @ mask @ model_d
    if icov is not None:
        icov = dt_icov @ mask @ icov

    if icov is None:
        e = DiagonalGaussianLikelihood(data=vis, inverse_covariance=wgt, nthreads=nthreads)
        return e @ model_d
    else:
        s0, s1 = "model data", "inverse covariance"
        e = ift.VariableCovarianceDiagonalGaussianLikelihood(vis, s0, s1, nthreads=nthreads)
        return e @ model_d.ducktape_left(s0) + icov.ducktape_left(s1)
