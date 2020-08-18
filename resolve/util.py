# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift


def my_assert(*conds):
    if not all(conds):
        raise RuntimeError


def my_asserteq(*args):
    for aa in args[1:]:
        if args[0] != aa:
            raise RuntimeError(args[0], aa)


def my_assert_isinstance(*args):
    args = list(args)
    cls = args.pop()
    for aa in args:
        if not isinstance(aa, cls):
            raise RuntimeError(aa, cls)


def compare_attributes(obj0, obj1, attribute_list):
    for a in attribute_list:
        compare = getattr(obj0, a) != getattr(obj1, a)
        if type(compare) is bool:
            if compare:
                return False
            continue
        if isinstance(compare, ift.MultiField):
            for vv in compare.val.values():
                if vv.any():
                    return False
            continue
        if compare.any():
            return False
    return True


def complex2float_dtype(dtype):
    if dtype == np.complex128:
        return np.float64
    if dtype == np.complex64:
        return np.float32
    raise RuntimeError
