# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import nifty7 as ift

from .minimization import MinimizationState
from .observation import Observation
from .util import my_assert_isinstance, my_asserteq


class Plotter:
    # TODO Residual plots
    def __init__(self, fileformat, directory):
        self._nifty, self._uvscatter = [], []
        self._f = fileformat
        self._dir = directory
        makedirs(self._dir, exist_ok=True)

    def add(self, name, operator, **kwargs):
        my_assert_isinstance(operator, ift.Operator)
        self._nifty.append({'operator': operator,
                            'title': str(name),
                            'kwargs': kwargs})

    def add_uvscatter(self, name, operator, observation):
        my_assert_isinstance(operator, ift.Operator)
        my_assert_isinstance(observation, Observation)
        my_asserteq(operator.target, observation.vis.domain)
        self._uvscatter.append({'operator': operator,
                                'observation': observation,
                                'title': str(name)})

    def plot(self, identifier, state):
        my_assert_isinstance(state, (ift.MultiField, MinimizationState))
        unit = 6
        for ii, obj in enumerate(self._nifty):
            op, kwargs = obj['operator'], obj['kwargs']
            if not set(op.domain.keys()) <= set(state.domain.keys()):
                continue
            direc = join(self._dir, obj['title'])
            makedirs(direc, exist_ok=True)
            fname = join(direc, f'{identifier}.{self._f}')

            if isinstance(state, MinimizationState) and len(state) > 0:
                if len(op.target) == 1 and isinstance(op.target[0], ift.RGSpace) and len(op.shape) == 1:
                    p = ift.Plot()
                    p.add([op.force(ss) for ss in state], **kwargs)
                    p.output(xsize=unit, ysize=unit, name=fname)
                else:
                    sc = ift.StatCalculator()
                    for ss in state:
                        sc.add(op.force(ss))
                    p = ift.Plot()
                    p.add(sc.mean, **kwargs)
                    p.add(sc.var.sqrt()/sc.mean)
                    p.output(nx=2, ny=1, xsize=2*unit, ysize=unit, name=fname)
            else:
                pos = state if isinstance(state, ift.MultiField) else state.mean
                ift.single_plot(op.force(pos), **kwargs, name=fname)

        for ii, obj in enumerate(self._uvscatter):
            op, obs = obj['operator'], obj['observation']
            if not set(op.domain.keys()) <= set(state.domain.keys()):
                continue
            direc = join(self._dir, obj['title'])
            makedirs(direc, exist_ok=True)
            fname = join(direc, f'{identifier}.{self._f}')

            withsamples = isinstance(state, MinimizationState) and len(state) > 0

            pos = state if isinstance(state, ift.MultiField) else state.mean
            uv = obs.effective_uv()
            u, v = uv[:, 0], uv[:, 1]
            uvwlen = obs.effective_uvwlen()
            ncols = 2 if withsamples else 1
            fig, axs = plt.subplots(obs.npol, 2*ncols, figsize=(2*ncols*unit, obs.npol*unit))
            axs = list(axs.ravel())
            if withsamples:
                sc = ift.StatCalculator()
                for ss in state:
                    sc.add(op.force(ss))
                weights = sc.mean
                relsd = sc.mean/sc.var.sqrt()
            else:
                weights = op.force(pos).val
            for axx in axs[::2]:
                axx.set_aspect('equal')
            for pol in range(obs.npol):
                axx = axs.pop(0)
                axx.set_title('Mean')
                sct = axx.scatter(u, v, c=weights[pol], s=1)
                fig.colorbar(sct, ax=axx)

                axx = axs.pop(0)
                axx.set_title('Mean')
                sct = axx.scatter(uvwlen, weights[pol], s=1)

                if withsamples:
                    axx = axs.pop(0)
                    axx.set_title('Rel. std dev')
                    sct = axx.scatter(u, v, c=relsd[pol], s=1)
                    fig.colorbar(sct, ax=axx)

                    axx = axs.pop(0)
                    axx.set_title('Rel. std dev')
                    sct = axx.scatter(uvwlen, relsd[pol], s=1)
            fig.savefig(fname)
            plt.close(fig)
