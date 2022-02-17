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
import pytest

import resolve as rve

from .common import list2fixture

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
    op0 = rve.polarization_matrix_exponential(dom, False)
    op1 = rve.polarization_matrix_exponential(dom, True)
    op2 = rve.polarization_matrix_exponential_mf2f({kk: restdom for kk in pdom.labels})

    loc = ift.from_random(op0.domain)

    ift.extra.check_operator(op0, loc, ntries=3)
    ift.extra.check_operator(op1, loc, ntries=3)
    ift.extra.assert_allclose(op0(loc), op1(loc))

    if len(restdom) == 3:
        op2 = op2 @ rve.MultiFieldStacker(dom, 0, pdom.labels).inverse
        ift.extra.check_operator(op2, loc, ntries=3)
        ift.extra.assert_allclose(op0(loc), op2(loc))


@pmp("pol", ("I", ["I", "Q", "U"], ["I", "Q", "U", "V"]))
def test_polarization(pol):
    dom = rve.PolarizationSpace(pol), rve.IRGSpace([0]), rve.IRGSpace([0]), ift.RGSpace([10, 20])
    op = rve.polarization_matrix_exponential(dom, False)
    pos = ift.from_random(op.domain)
    ift.extra.check_operator(op, pos, ntries=5)
    try:
        op_jax = rve.polarization_matrix_exponential(dom, True)

        assert op.domain is op_jax.domain
        assert op.target is op_jax.target

        ift.extra.assert_allclose(op(pos), op_jax(pos))
        ift.extra.check_operator(op_jax, pos, ntries=5)
    except ImportError:
        pass
