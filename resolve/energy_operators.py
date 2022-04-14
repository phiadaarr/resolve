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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import nifty8 as ift
import numpy as np

import resolve_support

from .cpp2py import Pybind11LikelihoodEnergyOperator
from .util import is_single_precision


def DiagonalGaussianLikelihood(
    data, inverse_covariance, mask=None, multiplicative=None, nthreads=1
):
    """Gaussian energy as NIFTy operator that is implemented in C++

    Parameters
    ----------
    data : Field

    inverse_covariance : Field
        Real valued. Needs to have same precision as `data`.

    mask : Field or None

    multiplicative : Field or None
        Field that is multiplied to the input before computing the residual.

    Note
    ----
    In contrast to the nifty interface of GaussianEnergy, this function takes a
    field as inverse_covariance.
    """
    if not ift.is_fieldlike(data):
        raise TypeError("data needs to be a Field")
    if not ift.is_fieldlike(inverse_covariance):
        raise TypeError("Inverse_covariance needs to be a Field")
    if data.domain != inverse_covariance.domain:
        raise ValueError(
            "data and inverse_covariance need to have the same domain",
            data.domain,
            inverse_covariance.domain,
        )
    dt = data.dtype
    if inverse_covariance.dtype != (np.float32 if is_single_precision(dt) else np.float64):
        raise ValueError("Precision of inverse_covariance does not match precision of data.")

    if dt == np.float64:
        f = resolve_support.DiagonalGaussianLikelihood_f8
    elif dt == np.float32:
        f = resolve_support.DiagonalGaussianLikelihood_f4
    elif dt == np.complex64:
        f = resolve_support.DiagonalGaussianLikelihood_c8
    elif dt == np.complex128:
        f = resolve_support.DiagonalGaussianLikelihood_c16
    else:
        raise TypeError("Dtype of data not supported. Supported dtypes: c8, c16.")

    if mask is None:
        mask_operator = ift.Operator.identity_operator(data.domain)
    else:
        foo = inverse_covariance.val.dtype
        mask = mask.val != 0.0
        # FIXME Somewhen fix this strange NIFTy convention
        mask_operator = ift.MaskOperator(ift.makeField(data.domain, ~mask))
        inverse_covariance = ift.makeField(data.domain, mask) * inverse_covariance
        assert inverse_covariance.val.dtype == foo

    if multiplicative is None:
        mask_operator_and_multiplicative = mask_operator
        multiplicative = np.broadcast_to(np.ones((1,), dtype=data.dtype), data.shape)
    else:
        assert multiplicative.domain == data.domain
        assert multiplicative.dtype == data.dtype
        mask_operator_and_multiplicative = mask_operator @ ift.makeOp(multiplicative)
        multiplicative = multiplicative.val

    if np.any(np.isnan(inverse_covariance.val)):
        raise ValueError("inverse_covariance must not contain NaN.")
    if np.any(np.isnan(data.val)):
        raise ValueError("data must not contain NaN.")

    return Pybind11LikelihoodEnergyOperator(
        data.domain,
        f(data.val, inverse_covariance.val, multiplicative, nthreads),
        nifty_equivalent=ift.GaussianEnergy(
            data=mask_operator(data),
            inverse_covariance=ift.makeOp(mask_operator(inverse_covariance), sampling_dtype=dt),
        ) @ mask_operator_and_multiplicative
    )


def VariableCovarianceDiagonalGaussianLikelihood(
    data, key_signal, key_log_inverse_covariance, mask, nthreads=1
):
    """Variable covariance Gaussian energy as NIFTy operator that is implemented in C++

    Parameters
    ----------
    data : Field

    key_signal : str

    key_log_inverse_covariance : str

    mask : Field or None

    nthreads : int

    Note
    ----
    In contrast to the nifty interface of VariableCovarianceGaussianEnergy,
    this function computes the residual as well and also takes the logarithm of
    the inverse covariance as input.
    """
    if not ift.is_fieldlike(data):
        raise TypeError("data needs to be a Field")
    dt = data.dtype

    if dt == np.float64:
        f = resolve_support.VariableCovarianceDiagonalGaussianLikelihood_f8
    elif dt == np.float32:
        f = resolve_support.VariableCovarianceDiagonalGaussianLikelihood_f4
    elif dt == np.complex64:
        f = resolve_support.VariableCovarianceDiagonalGaussianLikelihood_c8
    elif dt == np.complex128:
        f = resolve_support.VariableCovarianceDiagonalGaussianLikelihood_c16
    else:
        raise TypeError("Dtype of data not supported. Supported dtypes: c8, c16.")

    if mask is None:
        mask_operator = ift.Operator.identity_operator(data.domain)
        mask = np.ones(data.shape, np.uint8)
    else:
        mask = mask.val != 0.0
        # FIXME Somewhen fix this strange NIFTy convention
        mask_operator = ift.MaskOperator(ift.makeField(data.domain, ~mask))
        mask = mask.astype(np.uint8)

    flagged_data = mask_operator(data)

    return Pybind11LikelihoodEnergyOperator(
        {key_signal: data.domain, key_log_inverse_covariance: data.domain},
        f(data.val, key_signal, key_log_inverse_covariance, mask, nthreads),
        nifty_equivalent=ift.VariableCovarianceGaussianEnergy(
            flagged_data.domain, "residual", "icov", data.dtype
        )
        @ (
            (ift.Adder(flagged_data, neg=True) @ mask_operator)
            .ducktape_left("residual")
            .ducktape(key_signal)
            + (mask_operator @ ift.Operator.identity_operator(data.domain).exp()).ducktape_left("icov").ducktape(key_log_inverse_covariance)
        ),
    )
