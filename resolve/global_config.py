# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift

_wgridding = None
_epsilon = None
_nthreads = 1
_verbosity = False


def wgridding():
    if _wgridding is None:
        raise ValueError("Please set a value for wgridding via resolve.set_wgridding().")
    return _wgridding


def set_wgridding(val):
    global _wgridding
    _wgridding = bool(val)
    print(f"Set wgridding to {wgridding()}")


def epsilon():
    if _epsilon is None:
        raise ValueError("Please set a value for epsilon via resolve.set_epsilon().")
    return _epsilon


def set_epsilon(val):
    global _epsilon
    _epsilon = float(val)
    print(f"Set epsilon to {epsilon()}")


def nthreads():
    return _nthreads


def set_nthreads(val):
    global _nthreads
    _nthreads = int(val)
    ift.fft.set_nthreads(int(val))
    print(f"Set nthreads to {nthreads()}")


def set_verbosity(val):
    global _verbosity
    _verbosity = bool(val)
    if _verbosity:
        print("Activate verbose mode")
    else:
        print("Deactivate verbose mode")


def verbosity():
    return _verbosity
