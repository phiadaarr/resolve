# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift
import matplotlib.pyplot as plt
from .util import my_assert_isinstance


class Plotter:
    def __init__(self):
        raise NotImplementedError
        self._nifty = []
        self._hists = []

    def add(self, operator, name, **kwargs):
        my_assert_isinstance(operator, ift.Operator)
        self._nifty.append((operator, str(name), kwargs))

    def add_histogram(self, operator, **kwargs):
        my_assert_isinstance(operator, ift.Operator)
        self._hists.append((operator, kwargs))

    def add_uvscatter(self, operator, **kwargs):
        raise NotImplementedError

    def plot(self, name, position):
        for ii, (op, kwargs) in enumerate(self._ops):
            ift.single_plot(op.force(position), **kwargs[ii], name=f'{name}_{ii}.png')
        for jj, (op, kwargs) in enumerate(self._hists):
            plt.hist(op.force(position).ravel(), **kwargs)
            plt.savefig(f'{name}_{ii+1+jj}.png')
            plt.close()
