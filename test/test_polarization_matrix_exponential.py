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

from time import time

import nifty8 as ift
import numpy as np
import pytest

import resolve as rve

from .common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize

pdom = list2fixture([rve.PolarizationSpace("I"),
                     rve.PolarizationSpace(["I", "Q", "U"]),
                     rve.PolarizationSpace(["I", "Q", "U", "V"]),
                     ])

restdom = list2fixture([[ift.UnstructuredDomain(7)],
                        [ift.RGSpace([2, 3]), rve.IRGSpace([12., 13., 130])]
                        ])


def test_different_implementations(pdom, restdom):
    dom = tuple((pdom,)) + tuple(restdom)
    op0 = rve.polarization_matrix_exponential(dom)

    loc = ift.from_random(op0.domain)

    ift.extra.check_operator(op0, loc, ntries=3)

    if pdom.labels_eq(["I", "Q", "U", "V"]):
        op2 = rve.polarization_matrix_exponential_mf2f({kk: restdom for kk in pdom.labels})
        op2 = op2 @ rve.MultiFieldStacker(dom, 0, pdom.labels).inverse
        ift.extra.check_operator(op2, loc, ntries=3)
        ift.extra.assert_allclose(op0(loc), op2(loc))


@pmp("pol", ("I", ["I", "Q", "U"], ["I", "Q", "U", "V"]))
def test_polarization(pol):
    dom = rve.PolarizationSpace(pol), rve.IRGSpace([0]), rve.IRGSpace([0]), ift.RGSpace([10, 20])
    op = rve.polarization_matrix_exponential(dom)
    pos = ift.from_random(op.domain)
    ift.extra.check_operator(op, pos, ntries=5)


def test_polarization_matrix_exponential():
    nthreads = 1
    pdom = rve.PolarizationSpace(["I", "Q", "U", "V"])
    sdom = ift.RGSpace([2, 2])
    dom = rve.default_sky_domain(pdom=pdom, sdom=sdom)
    dom = {kk: dom[1:] for kk in pdom.labels}
    tgt = rve.default_sky_domain(pdom=pdom, sdom=sdom)
    mfs = rve.MultiFieldStacker(tgt, 0, tgt[0].labels)
    opold = rve.polarization_matrix_exponential(tgt) @ mfs
    op = rve.polarization_matrix_exponential_mf2f(opold.domain, nthreads)
    assert isinstance(op.domain, ift.MultiDomain)
    assert isinstance(op.target, ift.DomainTuple)
    rve.operator_equality(opold, op)
