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
            raise RuntimeError(f"{args[0]} != {aa}")


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
    elif isinstance(b, ift.Field):
        dom = b.domain
        b = b.val
    else:
        raise TypeError
    arr = np.divide(a, b, out=np.ones(dom.shape), where=b != 0)
    return ift.makeField(dom, arr)
