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

from .common import list2fixture, setup_function, teardown_function

dtype = list2fixture([np.complex64, np.complex128, np.float32, np.float64])
with_mask = list2fixture([False, True])
with_multiplicative = list2fixture([False, True])


def test_gaussian_energy(dtype, with_mask, with_multiplicative):
    dom = ift.UnstructuredDomain([15])
    mean = ift.from_random(dom, dtype=dtype)
    icov = ift.from_random(
        dom, dtype=(np.float32 if rve.is_single_precision(dtype) else np.float64)
    )
    icov = icov.exp()
    if with_mask:
        rng = np.random.default_rng(42)
        mask = (rng.uniform(0, 1, mean.shape) > 0.5).astype(np.uint8)
        mask = ift.makeField(mean.domain, mask)
    else:
        mask = None
    if with_multiplicative:
        multiplicative = ift.from_random(dom, dtype=dtype)
    else:
        multiplicative = None
    op1 = rve.DiagonalGaussianLikelihood(
        data=mean, inverse_covariance=icov, mask=mask, multiplicative=multiplicative, nthreads=1
    )
    op2 = rve.DiagonalGaussianLikelihood(
        data=mean, inverse_covariance=icov, mask=mask, multiplicative=multiplicative, nthreads=2
    )
    rve.operator_equality(op1.nifty_equivalent, op1, ntries=5, domain_dtype=dtype, rtol=1e-4)
    rve.operator_equality(op2.nifty_equivalent, op2, ntries=5, domain_dtype=dtype, rtol=1e-4)


def test_varcov_gaussian_energy(dtype, with_mask):
    dom = ift.UnstructuredDomain([4])
    mean = ift.from_random(dom, dtype=dtype)
    if with_mask:
        rng = np.random.default_rng(42)
        mask = (rng.uniform(0, 1, mean.shape) > 0.5).astype(np.uint8)
        mask = ift.makeField(mean.domain, mask)
    else:
        mask = None
    op1 = rve.VariableCovarianceDiagonalGaussianLikelihood(
        mean, "signal", "logicov", mask=mask, nthreads=1
    )
    op2 = rve.VariableCovarianceDiagonalGaussianLikelihood(
        mean, "signal", "logicov", mask=mask, nthreads=2
    )
    dt = {
        "signal": dtype,
        "logicov": rve.dtype_complex2float(dtype, force=True),
    }
    rve.operator_equality(op1.nifty_equivalent, op1, ntries=5, domain_dtype=dt, rtol=2e-5)
    rve.operator_equality(op2.nifty_equivalent, op2, ntries=5, domain_dtype=dt, rtol=2e-5)
