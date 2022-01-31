# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

import configparser
from os.path import join

import nifty8 as ift
import numpy as np
import pytest
import resolve as rve
from tempfile import TemporaryDirectory

pmp = pytest.mark.parametrize
np.seterr(all="raise")

obs = rve.ms2observations("/data/CYG-D-6680-64CH-10S.ms", "DATA", True, 0, polarizations="all")


@pmp("fname", ["cfg/cygnusa.cfg", "cfg/cygnusa_polarization.cfg", "cfg/mf.cfg",
               "cfg/cygnusa_mf.cfg"])
def test_build_multi_frequency_skymodel(fname):
    tmp = TemporaryDirectory()
    direc = tmp.name
    cfg = configparser.ConfigParser()
    cfg.read(fname)
    op, _ = rve.sky_model_diffuse(cfg["sky"], obs)
    out = op(ift.from_random(op.domain))

    rve.ubik_tools.field2fits(out, join(direc, "tmp.fits"))

    key1 = op.domain.keys()

    op, _ = rve.sky_model_points(cfg["sky"], obs)
    if op is not None:
        out = op(ift.from_random(op.domain))
        rve.ubik_tools.field2fits(out, join(direc, "tmp1.fits"))

        key2 = op.domain.keys()
        assert len(set(key1) & set(key2)) == 0
