# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

from os.path import join

import numpy as np
import pytest

import nifty8 as ift
import resolve as rve

pmp = pytest.mark.parametrize


@pmp("with_v", (False, True))
def test_polarization(with_v):
    dom = ift.RGSpace([10, 20])
    op = rve.polarization_matrix_exponential(dom, with_v, False)
    op_jax = rve.polarization_matrix_exponential(dom, with_v, True)

    assert op.domain is op_jax.domain
    assert op.target is op_jax.target
    pos = ift.from_random(op.domain)
    ift.extra.assert_allclose(op(pos), op_jax(pos))

    ift.extra.check_operator(op, pos, ntries=5)
    ift.extra.check_operator(op, pos, ntries=5)
