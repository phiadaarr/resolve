# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift

_wstacking = False
_epsilon = 1e-12
_nthreads = 1


def wstacking():
    return _wstacking


def set_wstacking(wstacking):
    print(f'Set wstacking to {wstacking}')
    global _wstacking
    _wstacking = bool(wstacking)


def epsilon():
    return _epsilon


def set_epsilon(epsilon):
    print(f'Set epsilon to {epsilon}')
    global _epsilon
    _epsilon = bool(epsilon)


def nthreads():
    return _nthreads


def set_nthreads(nthr):
    print(f'Set nthreads to {nthr}')
    global _nthreads
    _nthreads = int(nthr)
    ift.fft.set_nthreads(nthr)
