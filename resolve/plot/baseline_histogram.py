# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Author: Philipp Arras

import os
from ..constants import ARCMIN2RAD
from ..data.observation import unique_antennas

import nifty8 as ift
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def baseline_histogram(file_name, vis, observation, bins, weight=None):
    assert vis.domain == observation.vis.domain
    uvwlen = observation.effective_uvwlen().val
    pdom = vis.domain[0]

    if isinstance(bins, int):
        bins = np.linspace(uvwlen.min(), uvwlen.max(), num=bins, endpoint=True)
    assert np.min(uvwlen) >= np.min(bins)
    assert np.max(uvwlen) <= np.max(bins)
    assert np.all(np.sort(bins) == np.array(bins))

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, sharex=True)
    for pp in pdom.labels:
        ii = pdom.label2index(pp)
        luvwlen = uvwlen[ii]
        lvis = vis.val[ii]
        if np.iscomplexobj(lvis):
            lvis = np.abs(lvis)
            ax0.set_yscale("log")

        if weight is None:
            lweight = np.ones(lvis.shape)
        else:
            lweight = weight.val[ii]

        ys = []
        nys = []
        xs = []
        for ii in range(len(bins) - 1):
            mi, ma = bins[ii], bins[ii+1]
            inds = np.logical_and(luvwlen >= mi,
                                  luvwlen <= ma if ii == len(bins)-2 else
                                  luvwlen < ma)
            if np.sum(inds) == 0:
                continue
            weighted_average = np.mean(lvis[inds]* lvis[inds] * lweight[inds])
            xs.append(mi + 0.5*(ma-mi))
            ys.append(weighted_average)
            nys.append(np.sum(inds))
        xs = np.array(xs) * ARCMIN2RAD
        ax0.scatter(xs, ys, label=pp, alpha=0.5)
        ax1.scatter(xs, nys)
    ax1.set_ylabel("Number visibilities")
    ax1.set_xlabel("Effective baseline length [1/arcmin]")
    ax0.axhline(1, linestyle="--", alpha=0.5, color="k")
    ax0.set_ylim(*_red_chi_sq_limits(*ax0.get_ylim()))
    ax0.legend()

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def _red_chi_sq_limits(mi, ma):
    return min([mi, 1e-1]), max([ma, 1e1])


def antenna_matrix(file_name, vis, observation, weight=None, antenna_dct=None):
    if antenna_dct is not None:
        raise NotImplementedError

    ant1 = observation.ant1
    ant2 = observation.ant2
    pdom = vis.domain[0]
    n_antennas = max([np.max(ant1), np.max(ant2)])

    # Compute antenna distances
    coords = observation.antenna_coordinates
    lmat = np.empty((n_antennas, n_antennas))
    lmat[()] = np.nan
    for aa in range(n_antennas):
        for bb in range(aa+1):
            lmat[aa, bb] = lmat[bb, aa] = np.linalg.norm(coords[aa] - coords[bb])
    # /Compute antenna distances

    fig, axs = plt.subplots(nrows=observation.npol, ncols=3, figsize=(12, 4*observation.npol))
    axs = list(np.array(axs).ravel())
    for pp in pdom.labels:
        ii = pdom.label2index(pp)
        lvis = vis.val[ii]
        if np.iscomplexobj(lvis):
            lvis = np.abs(lvis)
        if weight is None:
            lweight = np.ones(lvis.shape)
        else:
            lweight = weight.val[ii]

        mat = np.empty((n_antennas, n_antennas))
        nmat = np.empty((n_antennas, n_antennas))
        mat[()] = np.nan
        nmat[()] = np.nan
        assert mat.shape == lmat.shape == nmat.shape
        xs = []
        for aa in range(n_antennas):
            for bb in range(aa+1):
                inds = np.logical_or(np.logical_and(ant1 == aa, ant2 == bb),
                                     np.logical_and(ant1 == bb, ant2 == aa))
                if np.sum(inds) == 0:
                    continue
                weighted_average = np.mean(lvis[inds]* lvis[inds] * lweight[inds])
                mat[aa, bb] = mat[bb, aa] = weighted_average
                nmat[aa, bb] = nmat[bb, aa] = np.sum(inds)
        axx = axs.pop(0)
        axx.set_xlabel("Antenna label")
        axx.set_ylabel("Antenna label")
        im = axx.matshow(mat,
                norm=LogNorm(*_red_chi_sq_limits(np.nanmin(mat), np.nanmax(mat))),
                cmap="seismic"
                )
        plt.colorbar(im, ax=axx, orientation="horizontal", label=f"Normalized residuals ({pp})")

        axx = axs.pop(0)
        axx.set_xlabel("Antenna label")
        axx.set_ylabel("Antenna label")
        im = axx.matshow(nmat, vmin=0)
        plt.colorbar(im, ax=axx, orientation="horizontal", label=f"# visibilities ({pp})")

        axx = axs.pop(0)
        axx.set_xlabel("Antenna label")
        axx.set_ylabel("Antenna label")
        im = axx.matshow(lmat, norm=LogNorm())
        plt.colorbar(im, ax=axx, orientation="horizontal", label="Baseline length [m]")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def visualize_weighted_residuals(obs_science, sl, iglobal, sky, weights, output_directory, io):
    from ..response_new import InterferometryResponse

    sky_mean = sl.average(sky)

    for ii, oo in enumerate(obs_science):
        # data weights
        model_vis = InterferometryResponse(oo, sky.target)(sky_mean)
        if io:
            dd = os.path.join(output_directory, f"normlized data residuals obs{ii} (data weights)")
            os.makedirs(dd, exist_ok=True)
            fname = os.path.join(dd, f"baseline_data_weights_iter{iglobal}_obs{ii}.png")
            baseline_histogram(fname, model_vis-oo.vis, oo, 100, weight=oo.weight)

            fname = os.path.join(dd, f"antenna_data_weights_iter{iglobal}_obs{ii}.png")
            antenna_matrix(fname, model_vis-oo.vis, oo, weight=oo.weight)
        # /data weights

        # learned weights
        if weights is None:
            continue
        dd = os.path.join(output_directory, f"normlized data residuals obs{ii} (learned weights)")
        weights_mean = sl.average(weights[ii])
        if io:
            os.makedirs(dd, exist_ok=True)
            fname = os.path.join(dd, f"baseline_model_weights_iter{iglobal}_obs{ii}.png")
            baseline_histogram(fname, model_vis-oo.vis, oo, 100, weight=weights_mean)

            fname = os.path.join(dd, f"antenna_model_weights_iter{iglobal}_obs{ii}.png")
            antenna_matrix(fname, model_vis-oo.vis, oo, weight=weights_mean)
        # /learned weights
