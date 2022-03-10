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
# Copyright(C) 2013-2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras

import nifty8 as ift
import numpy as np
import pytest

import resolve as rve


def list2fixture(lst):
    @pytest.fixture(params=lst)
    def myfixture(request):
        return request.param

    return myfixture


def operator_equality(op0, op1, ntries=20, domain_dtype=np.float64):
    dom = op0.domain
    assert op0.domain == op1.domain
    assert op0.target == op1.target
    rtol = 1e-5 if rve.is_single_precision(domain_dtype) else 1e-11
    for ii in range(ntries):
        loc = ift.from_random(dom, dtype=domain_dtype)
        res0 = op0(loc)
        res1 = op1(loc)
        ift.extra.assert_allclose(res0, res1, rtol=rtol)

        linloc = ift.Linearization.make_var(loc, want_metric=True)
        res0 = op0(linloc)
        res1 = op1(linloc)
        ift.extra.assert_allclose(res0.jac(0.23*loc), res1.jac(0.23*loc), rtol=rtol)
        if res0.metric is not None:
            ift.extra.assert_allclose(res0.metric(loc), res1.metric(loc), rtol=rtol)

        tgtloc = res0.jac(0.23*loc)
        res0 = op0(linloc).jac.adjoint(tgtloc)
        res1 = op1(linloc).jac.adjoint(tgtloc)
        ift.extra.assert_allclose(res0, res1, rtol=rtol)
    ift.extra.check_operator(op0, loc, ntries=ntries, tol=rtol)
    ift.extra.check_operator(op1, loc, ntries=ntries, tol=rtol)
