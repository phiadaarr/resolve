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
# Copyright(C) 2021-2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import nifty8 as ift
import numpy as np
import pytest

import resolve as rve

from .common import list2fixture, operator_equality

dtype = list2fixture([np.complex64, np.complex128, np.float32, np.float64])


def test_gaussian_energy(dtype):
    dom = ift.UnstructuredDomain([4])
    mean = ift.from_random(dom, dtype=dtype)
    icov = ift.from_random(dom,
                           dtype=(np.float32 if rve.is_single_precision(dtype) else np.float64))
    icov = icov.exp()
    op0 = ift.GaussianEnergy(data=mean, inverse_covariance=ift.makeOp(icov))
    op1 = rve.DiagonalGaussianLikelihood(data=mean, inverse_covariance=icov)
    operator_equality(op0, op1, ntries=5, domain_dtype=dtype)
    rve.set_nthreads(2)
    operator_equality(op0, op1, ntries=5, domain_dtype=dtype)
