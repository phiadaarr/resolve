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

from cpp2py import Pybind11Operator
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



pdom = rve.PolarizationSpace(["I", "Q", "U", "V"])
sdom = ift.RGSpace([10, 10])

dom = rve.default_sky_domain(pdom=pdom, sdom=sdom)
dom = {kk: dom[1:] for kk in pdom.labels}

tgt = rve.default_sky_domain(pdom=pdom, sdom=sdom)

opold = rve.polarization_matrix_exponential(tgt) @ rve.MultiFieldStacker(tgt, 0, tgt[0].labels)
op = Pybind11Operator(dom, tgt, resolvelib.PolarizationMatrixExponential(1))

operator_equality(opold, op)
exit()

print(f"Old implementation: {(time()-t0):.2f} s")
ntries = 50
for nthreads  in [8]:
#for nthreads  in range(1, 9):
    op = Pybind11Operator(dom, tgt, resolvelib.PolarizationMatrixExponential(nthreads))
    assert op.domain is opold.domain
    assert op.target is opold.target
    t0 = time()
    for _ in range(ntries):
        res = op(loc)
    print(f"Nthreads {nthreads}: {(time()-t0)/ntries:.2f} s")

    np.testing.assert_allclose(res0.val, res.val)
exit()
op(loc)
exit()

ift.extra.check_operator(op, loc)
