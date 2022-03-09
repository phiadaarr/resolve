# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np

_wgridding = None
_epsilon = None
_nthreads = 1
_verbosity = False
_double_precision = True


def wgridding():
    if _wgridding is None:
        raise ValueError("Please set a value for wgridding via resolve.set_wgridding().")
    return _wgridding


def set_wgridding(val):
    global _wgridding
    _wgridding = bool(val)


def epsilon():
    if _epsilon is None:
        raise ValueError("Please set a value for epsilon via resolve.set_epsilon().")
    return _epsilon


def set_epsilon(val):
    global _epsilon
    _epsilon = float(val)


def nthreads():
    return _nthreads


def set_nthreads(val):
    global _nthreads
    _nthreads = int(val)
    ift.set_nthreads(int(val))


def set_verbosity(val):
    global _verbosity
    _verbosity = bool(val)


def verbosity():
    return _verbosity
