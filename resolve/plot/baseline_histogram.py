# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Author: Philipp Arras

import os

import nifty8 as ift
import numpy as np

import resolve as rve


def baseline_histogram(file_name, vis, observation, bins, weight=None):
    import matplotlib.pyplot as plt

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
        xs = np.array(xs) * rve.ARCMIN2RAD
        ax0.scatter(xs, ys, label=pp, alpha=0.5)
        ax1.scatter(xs, nys)
    ax1.set_ylabel("Number visibilities")
    ax1.set_xlabel("Effective baseline length [1/arcmin]")
    ax0.axhline(1, linestyle="--", alpha=0.5, color="k")
    ymin, ymax = ax0.get_ylim()
    ax0.set_ylim([min([ymin, 1e-2]), max([ymax, 1e2])])
    ax0.legend()

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def visualize_weighted_residuals(obs_science, sl, iglobal, sky, weights, output_directory, io):
    from ..response_new import InterferometryResponse

    sky_mean = sl.average(sky)

    for ii, oo in enumerate(obs_science):
        # data weights
        model_vis = InterferometryResponse(oo, sky.target)(sky_mean)
        dd = os.path.join(output_directory, f"normlized data residuals obs{ii} (data weights)")
        if io:
            os.makedirs(dd, exist_ok=True)
            fname = os.path.join(dd, f"baseline_data_weights_iter{iglobal}_obs{ii}.png")
            baseline_histogram(fname, model_vis-oo.vis, oo, 100, weight=oo.weight)
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
        # /learned weights
