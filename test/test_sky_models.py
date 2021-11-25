# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

import configparser

import numpy as np
import pytest
import resolve as rve

pmp = pytest.mark.parametrize
np.seterr(all="raise")


def test_build_multi_frequency_skymodel():
    cfg = configparser.ConfigParser()
    cfg.read("test/mf_sky.cfg")
    rve.multi_frequency_sky(cfg["sky"])
