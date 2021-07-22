# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import numpy as np


def calibrator_spectrum(nu, src):
    """Return spectrum of calibration sources

    Parameters
    ----------
    nu : float or np.ndarray
        Frequency in Hz at which the spectrum is evaluated.
    src : str
        Name of the calibrator.


    Returns
    -------
    float or np.ndarray : calibrator spectrum

    Note
    ----
    Currently only the calibrator J1939_6342 is supported.
    """
    if src != "J1939_6342":
        raise ValueError("Source not in catalogue.")
    I0 = 14.907688923070978
    coefficients = [-0.19473726758824528, -0.6012047948505569,
                    0.11417312472848717, 9.069178705870204e-08]
    nu0 = 1.4e9
    w = nu/nu0
    logw = np.log10(w)
    expon = reduce(add, [coeff*(logw**ii) for ii, coeff in enumerate(coefficients)])
    if isinstance(nu, np.ndarray):
        assert expon.shape == nu.shape
    return I0 * (w**expon)
