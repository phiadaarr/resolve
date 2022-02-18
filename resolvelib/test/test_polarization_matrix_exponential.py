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

import resolve as rve
import nifty8 as ift
import resolvelib
import numpy as np

from time import time


def operator_equality(op0, op1, ntries=20):
    dom = op0.domain
    assert op0.domain == op1.domain
    assert op0.target == op1.target
    for ii in range(ntries):
        loc = ift.from_random(dom)
        res0 = op0(loc)
        res1 = op1(loc)
        ift.extra.assert_allclose(res0, res1)

        linloc = ift.Linearization.make_var(loc)
        res0 = op0(linloc).jac(0.23*loc)
        res1 = op1(linloc).jac(0.23*loc)
        ift.extra.assert_allclose(res0, res1)
    ift.extra.check_operator(op0, loc, ntries=ntries)
    ift.extra.check_operator(op1, loc, ntries=ntries)


def test_polarization_matrix_exponential():
    nthreads = 1
    pdom = rve.PolarizationSpace(["I", "Q", "U", "V"])
    sdom = ift.RGSpace([2, 2])
    dom = rve.default_sky_domain(pdom=pdom, sdom=sdom)
    dom = {kk: dom[1:] for kk in pdom.labels}
    tgt = rve.default_sky_domain(pdom=pdom, sdom=sdom)
    mfs = rve.MultiFieldStacker(tgt, 0, tgt[0].labels)
    opold = rve.polarization_matrix_exponential(tgt) @ mfs
    op = resolvelib.PolarizationMatrixExponential(opold.domain, nthreads)
    assert isinstance(op.domain, ift.MultiDomain)
    assert isinstance(op.target, ift.DomainTuple)
    operator_equality(opold, op)
