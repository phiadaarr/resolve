# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Author: Philipp Arras

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
    ax0.set_ylim([1e-1, 1e3])
    ax0.legend()

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
