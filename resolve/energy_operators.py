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

import resolvelib

from .cpp2py import Pybind11LikelihoodEnergyOperator
from .global_config import nthreads
from .util import is_single_precision


def DiagonalGaussianLikelihood(data, inverse_covariance):
    """Gaussian energy as NIFTy operator that is implemented in C++

    Parameters
    ----------
    data : Field

    inverse_covariance : Field
        Real valued. Needs to have same precision as `data`.

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
        raise ValueError("data and inverse_covariance need to have the same domain",
                         data.domain, inverse_covariance.domain)
    dt = data.dtype
    if inverse_covariance.dtype != (np.float32 if is_single_precision(dt) else np.float64):
        raise ValueError("Precision of inverse_covariance does not match precision of data.")

    if dt == np.float64:
        f = resolvelib.DiagonalGaussianLikelihood_f8
    elif dt == np.float32:
        f = resolvelib.DiagonalGaussianLikelihood_f4
    elif dt == np.complex64:
        f = resolvelib.DiagonalGaussianLikelihood_c8
    elif dt == np.complex128:
        f = resolvelib.DiagonalGaussianLikelihood_c16
    else:
        raise TypeError("Dtype of data not supported. Supported dtypes: c8, c16.")

    def draw_sample(from_inverse=False):
        return ift.makeOp(inverse_covariance, sampling_dtype=dt).draw_sample(from_inverse)

    trafo = ift.makeOp(inverse_covariance).get_sqrt()  # NOTE This is not implemented in C++ yet
    return Pybind11LikelihoodEnergyOperator(
            data.domain,
            f(data.val, inverse_covariance.val, nthreads()),
            lambda x: draw_sample,
            get_transformation=lambda: (dt, trafo),
            data_residual=ift.Adder(data, neg=True),  # NOTE This is not implemented in C++ yet
            sqrt_data_metric=lambda x: trafo,
            )
