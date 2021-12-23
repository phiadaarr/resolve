# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import pytest
import resolve as rve

pmp = pytest.mark.parametrize


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
