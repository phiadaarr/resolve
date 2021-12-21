# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

import configparser

import numpy as np
import nifty8 as ift
import pytest
import resolve as rve

pmp = pytest.mark.parametrize
np.seterr(all="raise")


def test_build_multi_frequency_skymodel():
    cfg = configparser.ConfigParser()
    cfg.read("test/mf_sky.cfg")
    op, _ = rve.sky_model(cfg["sky"])
    op(ift.from_random(op.domain))
    assert op.target[0].size == 1
    assert op.target[1].size == 1
    assert op.target[2].size == 10

    cfg.read("test/mf_sky_wiener_process.cfg")
    op, _ = rve.sky_model(cfg["sky"])
    op(ift.from_random(op.domain))

    cfg["sky"]["stokesI diffuse wp asperity mean"] = "0.1"
    cfg["sky"]["stokesI diffuse wp asperity stddev"] = "0.1"
    op, _ = rve.sky_model(cfg["sky"])
    op(ift.from_random(op.domain))

    cfg["sky"]["freq"] = "data"
    op, _ = rve.sky_model(cfg["sky"])
    op(ift.from_random(op.domain))
