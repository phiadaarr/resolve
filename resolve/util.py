# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

import matplotlib.pyplot as plt
import numpy as np

import nifty8 as ift


def my_assert(*conds):
    if not all(conds):
        raise RuntimeError


def my_asserteq(*args):
    for aa in args[1:]:
        if args[0] != aa:
            raise RuntimeError(f"{args[0]} != {aa}")


def my_assert_isinstance(*args):
    args = list(args)
    cls = args.pop()
    for aa in args:
        if not isinstance(aa, cls):
            raise RuntimeError(aa, cls)


def compare_attributes(obj0, obj1, attribute_list):
    return all(_fancy_equal(getattr(obj0, a), getattr(obj1, a))
               for a in attribute_list)


def _fancy_equal(o1, o2):
    if not _types_equal(o1, o2):
        return False

    # Turn MultiField into dict
    if isinstance(o1, ift.MultiField):
        o1, o2 = o1.val, o2.val

    # Compare dicts
    if isinstance(o1, dict):
        return _deep_equal(o1, o2)

    # Compare simple objects and np.ndarrays
    return _compare_simple_or_array(o1, o2)


def _deep_equal(a, b):
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise TypeError

    if a.keys() != b.keys():
        return False

    return all(_compare_simple_or_array(a[kk], b[kk]) for kk in a.keys())


def _compare_simple_or_array(a, b):
    equal = a == b
    try:
        return bool(equal)
    except ValueError:
        return equal.all()


def _types_equal(a, b):
    return type(a) == type(b)


class Reshaper(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.makeField(self._tgt(mode), x.val.reshape(self._tgt(mode).shape))


def divide_where_possible(a, b):
    # Otherwise one
    if isinstance(a, ift.Field):
        dom = a.domain
        a = a.val
        dtype = a.dtype
        if isinstance(b, ift.Field):
            my_asserteq(b.dtype, a.dtype)
    elif isinstance(b, ift.Field):
        dom = b.domain
        b = b.val
        dtype = b.dtype
    else:
        raise TypeError
    arr = np.divide(a, b, out=np.ones(dom.shape, dtype=dtype), where=b != 0)
    return ift.makeField(dom, arr)


def imshow(arr, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    return ax.imshow(arr.T, origin="lower", **kwargs)


def rows_to_baselines(antenna_positions, data_field):
    ua = antenna_positions.unique_antennas()
    my_assert(np.all(antenna_positions.ant1 < antenna_positions.ant2))

    res = {}
    for iant1 in range(len(ua)):
        for iant2 in range(iant1+1, len(ua)):
            res[f"{iant1}-{iant2}"] = antenna_positions.extract_baseline(iant1, iant2, data_field)
    return res


class DomainChanger(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        return ift.makeField(self._tgt(mode), x.val)
