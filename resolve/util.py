# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras


def my_assert(cond):
    if not cond:
        raise RuntimeError


def compare_attributes(obj0, obj1, attribute_list):
    for a in attribute_list:
        compare = getattr(obj0, a) != getattr(obj1, a)
        if type(compare) is bool:
            if compare:
                return False
            continue
        if compare.any():
            return False
    return True
