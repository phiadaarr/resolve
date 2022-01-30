# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift

obs = rve.ms2observations("/data/CYG-ALL-2052-2MHZ.ms", "DATA", True, 0)[0]


def baseline_histogram(file_name, vis, observation, bins, weight=None):
    import matplotlib.pyplot as plt

    assert vis.domain == observation.vis.domain
    uvwlen = obs.effective_uvwlen().val
    pdom = vis.domain[0]

    assert np.min(uvwlen) >= np.min(bins)
    assert np.max(uvwlen) < np.max(bins)
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
            inds = np.logical_and(luvwlen >= mi, luvwlen < ma)
            if np.sum(inds) == 0:
                continue
            weighted_average = np.average(lvis[inds], weights=lweight[inds])
            xs.append(mi + 0.5*(ma-mi))
            ys.append(weighted_average)
            nys.append(np.sum(inds))
        xs = np.array(xs) * rve.ARCMIN2RAD
        ax0.scatter(xs, ys, label=pp, alpha=0.5)
        ax1.scatter(xs, nys)
    ax1.set_ylabel("Number visibilities")
    ax1.set_xlabel("Effective baseline length [1/arcmin]")
    ax0.legend()

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()



mi = obs.effective_uvwlen().val.min()
ma = obs.effective_uvwlen().val.max()
bins = list(ift.PowerSpace.linear_binbounds(100, mi, ma)) + [np.inf]
baseline_histogram("test.png", obs.vis, obs, bins, weight=obs.weight)
