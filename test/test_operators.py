# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

from os.path import join

import numpy as np
import pytest

import nifty8 as ift
import resolve as rve

pmp = pytest.mark.parametrize
np.seterr(all="raise")


def test_reshaper():
    dom = ift.UnstructuredDomain([3]), ift.RGSpace([5])
    tgt = ift.RGSpace([15])
    op = rve.DomainChangerAndReshaper(dom, tgt)
    ift.extra.check_linear_operator(op)
