# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift
from .util import my_assert_isinstance


class Plotter:
    def __init__(self):
        self._ops = []
        self._kwargs = []

    def add(self, operator, **kwargs):
        my_assert_isinstance(operator, ift.Operator)
        self._ops.append(operator)
        self._kwargs.append(kwargs)

    def plot(self, name, position):
        for ii in range(len(self._ops)):
            ift.single_plot(self._ops[ii].force(position), **self._kwargs[ii], name=f'{name}_{ii}.png')
