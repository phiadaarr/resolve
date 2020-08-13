# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift

_wstacking = False
_epsilon = 1e-12
_nthreads = 1


def wstacking():
    return _wstacking


def set_wstacking(val):
    global _wstacking
    _wstacking = bool(val)
    print(f'Set wstacking to {wstacking()}')


def epsilon():
    return _epsilon


def set_epsilon(val):
    global _epsilon
    _epsilon = float(val)
    print(f'Set epsilon to {epsilon()}')


def nthreads():
    return _nthreads


def set_nthreads(val):
    global _nthreads
    _nthreads = int(val)
    ift.fft.set_nthreads(int(val))
    print(f'Set nthreads to {nthreads()}')
