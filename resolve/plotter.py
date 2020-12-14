# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import os
from os.path import join, splitext

import matplotlib.pyplot as plt
import numpy as np

import nifty7 as ift

from .minimization import MinimizationState
from .mpi import onlymaster
from .util import my_assert_isinstance, my_asserteq

UNIT = 6


@onlymaster
def makedirs(*args, **kwargs):
    from os import makedirs

    makedirs(*args, **kwargs)


class Plotter:
    def __init__(self, fileformat, directory):
        self._nifty, self._calib, self._hist = [], [], []
        self._f = fileformat
        self._dir = directory
        makedirs(self._dir, exist_ok=True)

    @onlymaster
    def add_npy(self, name, operator, nsamples=0):
        """Save output of operator as npy files.

        This function may be useful for saving results of a resolve run and
        plotting them later with dedicated plotting scripts.

        If the plotter gets samples during plotting, the posterior mean and the
        posterior standard deviation are saved. Additionally, a number of
        posterior samples are saved. If no samples are available, only one npy
        array will be saved.

        Parameters
        ----------
        name : str
            Naming of the output directory and files.
        operator : Operator
            The output of this operator is saved
        nsamples : int
            Determines how many posterior samples shall be saved additionally to
            the posterior mean.

        """
        raise NotImplementedError

    @onlymaster
    def add(self, name, operator, **kwargs):
        my_assert_isinstance(operator, ift.Operator)
        self._nifty.append({"operator": operator, "title": str(name), "kwargs": kwargs})

    @onlymaster
    def add_calibration_solution(self, name, operator):
        my_assert_isinstance(operator, ift.Operator)
        self._calib.append({"title": str(name), "operator": operator})

    @onlymaster
    def add_histogram(self, name, operator):
        my_assert_isinstance(operator, ift.Operator)
        self._hist.append({"title": str(name), "operator": operator})

    @onlymaster
    def plot(self, identifier, state):
        my_assert_isinstance(state, (ift.MultiField, MinimizationState))
        for ii, obj in enumerate(self._nifty):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            print(ii, obj)
            try:
                _plot_nifty(state, op, obj["kwargs"], fname)
            except:
                pass
        for ii, obj in enumerate(self._calib):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            _plot_calibration(state, op, fname)
        for ii, obj in enumerate(self._hist):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            _plot_histograms(state, fname, 20, 0.5, postop=op)
        mydir = join(self._dir, "zzz_latent")
        makedirs(mydir, exist_ok=True)
        _plot_histograms(state, join(mydir, f"{identifier}.{self._f}"), 5, 0.5)

    @onlymaster
    def _plot_init(self, obj, state, identifier):
        op = obj["operator"]
        if not set(op.domain.keys()) <= set(state.domain.keys()):
            return None, None
        direc = join(self._dir, obj["title"])
        makedirs(direc, exist_ok=True)
        fname = join(direc, f"{identifier}.{self._f}")
        return op, fname

    @property
    def directory(self):
        return self._dir


class MfPlotter(Plotter):
    def __init__(self, fileformat, directory):
        super(MfPlotter, self).__init__(fileformat, directory)
        self._spectra = []
        self._mf = []
        self._multi1d = []

    @onlymaster
    def add_spectra(self, name, operator, directions):
        """
        Parameters
        ----------
        directions: array, shape (n, 2)
            In units of the space.
        """
        my_assert_isinstance(operator, ift.Operator)
        # FIXME Write interface checks
        sdom = operator.target[1]
        directions = np.round(np.array(directions) / np.array(sdom.distances)).astype(
            int
        )
        self._spectra.append(
            {"title": str(name), "operator": operator, "directions": directions}
        )

    @onlymaster
    def add_multiple2d(self, name, operator, directions=None, movie_length=2):
        """
        Parameters
        ----------
        directions: array, shape (n, 2)
            In units of the space.
        """
        my_assert_isinstance(operator, ift.Operator)
        # FIXME Write interface checks
        self._mf.append(
            {
                "title": str(name),
                "operator": operator,
                "directions": directions,
                "movie_length": movie_length,
            }
        )

    @onlymaster
    def add_multiple1d(self, name, operator, **kwargs):
        my_assert_isinstance(operator, ift.Operator)
        my_asserteq(len(operator.target.shape), 2)
        self._multi1d.append(
            {"operator": operator, "title": str(name), "kwargs": kwargs}
        )

    @onlymaster
    def plot(self, identifier, state):
        super(MfPlotter, self).plot(identifier, state)

        for ii, obj in enumerate(self._spectra):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            _plot_spectra(state, op, fname, obj["directions"])

        for ii, obj in enumerate(self._mf):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            _plot_mf(state, op, fname, obj["directions"], obj["movie_length"])

        for ii, obj in enumerate(self._multi1d):
            op, fname = self._plot_init(obj, state, identifier)
            if fname is None:
                continue
            _plot_multi_oned(state, op, obj["kwargs"], fname)


def _plot_nifty(state, op, kwargs, fname):
    if isinstance(state, MinimizationState) and len(state) > 0:
        tgt = op.target
        if (
            len(tgt) == 1
            and isinstance(tgt[0], (ift.RGSpace, ift.PowerSpace))
            and len(tgt.shape) == 1
        ):
            p = ift.Plot()
            p.add([op.force(ss) for ss in state], **kwargs)
            p.output(xsize=UNIT, ysize=UNIT, name=fname)
        else:
            sc = ift.StatCalculator()
            for ss in state:
                sc.add(op.force(ss))
            p = ift.Plot()
            p.add(sc.mean, **kwargs)
            p.add(sc.var.sqrt() / sc.mean)
            p.output(nx=2, ny=1, xsize=2 * UNIT, ysize=UNIT, name=fname)
    else:
        pos = state if isinstance(state, ift.MultiField) else state.mean
        ift.single_plot(op.force(pos), **kwargs, name=fname)


def _plot_multi_oned(state, op, kwargs, fname):
    if isinstance(state, MinimizationState) and len(state) > 0:
        raise NotImplementedError
        p = ift.Plot()
        p.add([op.force(ss) for ss in state], **kwargs)
        p.output(xsize=UNIT, ysize=UNIT, name=fname)
    else:
        pos = state if isinstance(state, ift.MultiField) else state.mean
        ift.single_plot(op.force(pos), **kwargs, name=fname)


def _plot_calibration(state, op, fname):
    npol, nants, _, nfreq = op.target.shape
    if nfreq != 1:
        raise NotImplementedError
    lst = [op.force(ll) for ll in _generalstate2list(state)]
    xs = np.arange(op.target[2].shape[0])
    if isinstance(op.target[2], ift.RGSpace):
        xs = xs * op.target[2].distances[0] / 3600
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(4 * UNIT, npol * UNIT))
    axs = list(axs.ravel())
    for ii in range(npol):
        axx = axs[ii]
        axx.set_title(f"Polarization {ii}")
        colors = plt.cm.viridis(np.linspace(0, 1, nants))
        for jj in range(nants):
            for ll in lst:
                axx.plot(xs, ll.val[ii, jj], alpha=0.3, color=colors[jj])
    axs[1].set_xlabel("Time [h]")
    fig.savefig(fname)
    plt.close(fig)


def _generalstate2list(state):
    if isinstance(state, (ift.Field, ift.MultiField)):
        return [state]
    elif isinstance(state, MinimizationState):
        if len(state) == 0:
            return [state.mean]
        else:
            return state
    raise TypeError


def _plot_histograms(state, fname, lim, width, postop=None):
    lst = _generalstate2list(state)
    if postop is not None:
        lst0 = postop.force(lst[0])
        lst = (postop.force(ff) for ff in lst)
        dom = postop.target
    else:
        lst0 = lst[0]
        dom = state.domain

    if isinstance(dom, ift.DomainTuple):
        iscplx = {"": np.iscomplexobj(lst0.val)}
    else:
        iscplx = {kk: np.iscomplexobj(lst0[kk]) for kk in dom.keys()}

    N = sum(iscplx.values()) * 2 + sum(not aa for aa in iscplx.values())
    n1 = int(np.ceil(np.sqrt(N)))
    n2 = int(np.ceil(N / n1))

    if n1 == n2 == 1:
        fig = plt.figure(figsize=(UNIT / 2 * n1, UNIT / 2 * n2))
        axs = [plt.gca()]
    else:
        fig, axs = plt.subplots(n2, n1, figsize=(UNIT / 2 * n1, UNIT * 3 / 4 * n2))
        axs = list(axs.ravel())

    for kk, cplx in iscplx.items():
        if isinstance(dom, ift.MultiDomain):
            lstA = np.array([ss[kk].val for ss in lst])
        else:
            lstA = np.array([ss.val for ss in lst])
        if cplx:
            lstC = {"real": lstA.real, "imag": lstA.imag}
        else:
            lstC = {"": lstA}
        for ll, lstD in lstC.items():
            axx = axs.pop(0)
            lstD = lstD.ravel()
            if lstD.size == 1:
                axx.axvline(lstD[0])
            else:
                axx.hist(
                    lstD.clip(-lim, lim - width),
                    alpha=0.5,
                    density=True,
                    bins=np.arange(-lim, lim + width, width),
                )
            xs = np.linspace(-5, 5)
            axx.set_yscale("log")
            axx.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi))
            redchisq = (
                np.sum((np.abs(lstD) if np.iscomplexobj(lstD) else lstD) ** 2)
                / lstD.size
            )
            axx.set_title(f"{kk} {ll} {redchisq:.1f}")
    plt.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def _plot_spectra(state, op, name, directions):
    fld = op.force(state.mean)
    mi, ma = np.min(fld.val), np.max(fld.val)

    fdom, dom = fld.domain
    freqs = np.array(fdom.coordinates) * 1e-6

    plt.figure()
    for indx, indy in directions:
        plt.plot(freqs, fld.val[:, indx, indy])
    plt.ylim([mi, ma])
    plt.xlabel("MHz")
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def _plot_mf(state, op, name, directions, movie_length):
    # FIXME Write proper plotting routine which includes also directions and uncertainties
    fld = op.force(state.mean)

    fdom, dom = fld.domain
    freqs = np.array(fdom.coordinates)
    nfreq = fld.shape[0]
    my_asserteq(nfreq, len(freqs))
    mi, ma = np.min(fld.val), np.max(fld.val)

    name = splitext(name)[0]
    for ii in range(nfreq):
        ift.single_plot(
            ift.makeField(dom, fld.val[ii]),
            title=f"{freqs[ii]*1e-6:.0f} MHz",
            vmin=mi,
            vmax=ma,
            cmap="inferno",
            name=f"{name}_{ii:04.0f}.png",
        )

    if movie_length > 0:
        framerate = nfreq / float(movie_length)
        os.system(
            f"ffmpeg -framerate {framerate} -i {name}_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 -y {name}.mp4"
        )
        for ii in range(nfreq):
            os.remove(f"{name}_{ii:04.0f}.png")
