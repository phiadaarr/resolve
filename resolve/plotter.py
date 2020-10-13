# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

from os import makedirs
from os.path import join

import matplotlib.pyplot as plt
import numpy as np

import nifty7 as ift

from .minimization import MinimizationState
from .observation import Observation
from .util import my_assert_isinstance, my_asserteq

UNIT = 6


class Plotter:
    # TODO Residual plots
    def __init__(self, fileformat, directory):
        self._nifty, self._uvscatter, self._calib = [], [], []
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

    def add_calibration_solution(self, name, operator):
        my_assert_isinstance(operator, ift.Operator)
        self._calib.append({'title': str(name), 'operator': operator})

    def plot(self, identifier, state):
        my_assert_isinstance(state, (ift.MultiField, MinimizationState))
        for ii, obj in enumerate(self._nifty):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            _plot_nifty(state, op, obj['kwargs'], fname)
        for ii, obj in enumerate(self._uvscatter):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            obs = obj['observation']
            _plot_uvscatter(state, op, obs, fname)
        for ii, obj in enumerate(self._calib):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            _plot_calibration(state, op, fname)
        mydir = join(self._dir, 'zzz_latent')
        makedirs(mydir, exist_ok=True)
        _plot_latent_histograms(state, join(mydir, f'{identifier}.{self._f}'))

    def _plot_init(self, obj, state, identifier):
        op = obj['operator']
        if not set(op.domain.keys()) <= set(state.domain.keys()):
            return None, None
        direc = join(self._dir, obj['title'])
        makedirs(direc, exist_ok=True)
        fname = join(direc, f'{identifier}.{self._f}')
        return op, fname

    @property
    def directory(self):
        return self._dir


def _plot_nifty(state, op, kwargs, fname):
    if isinstance(state, MinimizationState) and len(state) > 0:
        if len(op.target) == 1 and isinstance(op.target[0], ift.RGSpace) and len(op.shape) == 1:
            p = ift.Plot()
            p.add([op.force(ss) for ss in state], **kwargs)
            p.output(xsize=UNIT, ysize=UNIT, name=fname)
        else:
            sc = ift.StatCalculator()
            for ss in state:
                sc.add(op.force(ss))
            p = ift.Plot()
            p.add(sc.mean, **kwargs)
            p.add(sc.var.sqrt()/sc.mean)
            p.output(nx=2, ny=1, xsize=2*UNIT, ysize=UNIT, name=fname)
    else:
        pos = state if isinstance(state, ift.MultiField) else state.mean
        ift.single_plot(op.force(pos), **kwargs, name=fname)


def _plot_calibration(state, op, fname):
    withsamples = isinstance(state, MinimizationState) and len(state) > 0
    pos = state if isinstance(state, ift.MultiField) else state.mean
    if withsamples:
        raise NotImplementedError
    npol, nants, _, nfreq = op.target.shape
    if nfreq != 1:
        raise NotImplementedError
    xs = np.arange(op.target[2].shape[0])
    if isinstance(op.target[2], ift.RGSpace):
        xs = xs*op.target[2].distances[0]/3600
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4*UNIT, npol*UNIT))
    axs = list(axs.ravel())
    for ii in range(npol):
        axx = axs[ii]
        axx.set_title(f'Polarization {ii}')
        colors = plt.cm.viridis(np.linspace(0, 1, nants))
        for jj in range(nants):
            ys = op.force(pos).val[ii, jj]
            axx.plot(xs, ys, alpha=0.3, color=colors[jj])
    axs[1].set_xlabel('Time [h]')
    fig.savefig(fname)
    plt.close(fig)


def _plot_uvscatter(state, op, obs, fname):
    withsamples = isinstance(state, MinimizationState) and len(state) > 0
    pos = state if isinstance(state, ift.MultiField) else state.mean
    uv = obs.effective_uv()
    u, v = uv[:, 0], uv[:, 1]
    uvwlen = obs.effective_uvwlen()
    ncols = 2 if withsamples else 1
    fig, axs = plt.subplots(obs.npol, 2*ncols, figsize=(2*ncols*UNIT, obs.npol*UNIT))
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


def _plot_latent_histograms(state, fname):
    pos = state if isinstance(state, ift.MultiField) else state.mean
    N = len(state.domain.keys())
    n1 = int(np.ceil(np.sqrt(N)))
    n2 = int(np.ceil(N/n1))
    fig, axs = plt.subplots(n1, n2, figsize=(UNIT/2*n1, UNIT/2*n2))
    axs = list(axs.ravel())
    for kk in pos.keys():
        axx = axs.pop(0)
        if isinstance(state, ift.MultiField):
            arr = state[kk].val
        elif len(state) == 0:
            arr = state.mean[kk].val
        else:
            arr = np.array([ss[kk].val for ss in state])
        if arr.size == 1:
            axx.axvline(arr.ravel()[0])
        else:
            mi, ma, width = np.min(arr), np.max(arr), 0.3
            axx.hist(arr.ravel(), alpha=0.5, density=True,
                     bins=np.arange(mi, ma + width, width))
        xs = np.linspace(-5, 5)
        axx.plot(xs, np.exp(-xs**2/2)/np.sqrt(2*np.pi))
        axx.set_title(f'{kk}')
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)
