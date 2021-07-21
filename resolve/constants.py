# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np
from scipy.constants import speed_of_light

ARCMIN2RAD = np.pi / 60 / 180
AS2RAD = ARCMIN2RAD / 60
DEG2RAD = np.pi / 180
SPEEDOFLIGHT = speed_of_light


def str2rad(s):
    """Convert string of number and unit to radian.

    Support the following units: muas mas as amin deg rad.

    Parameters
    ----------
    s : str
        TODO

    """
    c = {
        "muas": AS2RAD * 1e-6,
        "mas": AS2RAD * 1e-3,
        "as": AS2RAD,
        "amin": ARCMIN2RAD,
        "deg": DEG2RAD,
        "rad": 1,
    }
    keys = list(c.keys())
    keys.sort(key=len)
    for kk in reversed(keys):
        nn = -len(kk)
        unit = s[nn:]
        if unit == kk:
            return float(s[:nn]) * c[kk]
    raise RuntimeError("Unit not understood")
