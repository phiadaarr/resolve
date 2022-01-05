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

obs = rve.ms2observations("/data/CYG-D-6680-64CH-10S.ms", "DATA", True, 0, polarizations="all")


@pmp("fname", ["cfg/cygnusa.cfg", "cfg/cygnusa_polarization.cfg", "cfg/mf.cfg",
               "cfg/cygnusa_mf.cfg"])
def test_build_multi_frequency_skymodel(fname):
    cfg = configparser.ConfigParser()
    cfg.read(fname)
    op, _, _ = rve.sky_model(cfg["sky"], obs)
    op(ift.from_random(op.domain))
