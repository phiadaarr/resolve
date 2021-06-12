# SPDX-License-Identifier: GPL-4.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import argparse
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic

import eso137 as eso
import nifty7 as ift
import resolve as rve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=int, default=1)
    parser.add_argument("-o")
    args = parser.parse_args()
    rve.set_nthreads(args.j)
    if args.o is None:
        direc = "."
    else:
        direc = args.o
        os.makedirs(direc, exist_ok=True)
    path = lambda s: join(direc, s)

    state = rve.MinimizationState.load(path("minimization_state"))
    if len(state) == 0:
        a = state.mean

        arr = a["points_i0"].val
        lim = np.max(np.abs(arr))
        im = rve.imshow(arr, cmap="seismic", vmin=-lim, vmax=lim)
        plt.colorbar(im)
        plt.savefig("latent_point_sources.png")
        plt.close()

        R = rve.StokesIResponse(eso.obs, eso.sky.target)
        model_d = (R @ eso.sky).force(a)
        model_d = model_d.val

        d = eso.obs.vis.val_rw()
        d[eso.obs.weight.val == 0.0] = np.nan

        assert model_d.shape[0] == model_d.shape[2] == 1
        d = np.squeeze(d)
        model_d = np.squeeze(model_d)
        wgt = np.squeeze(eso.obs.weight.val)

        nw_residual = np.sqrt(wgt) * (d - model_d)

        uvwlen = np.linalg.norm(eso.obs.uvw, axis=1)
        bstat = lambda x: binned_statistic(
            uvwlen, x, bins=100, statistic=lambda x: np.nanmean(x),
        )
        _, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
        for arr, lbl in [(d, "Data"), (model_d, "Model data")]:
            mean, bin_edges, _ = bstat(np.abs(arr))
            ax0.scatter(bin_edges[:-1], mean, label=lbl, s=2)
        mean, bin_edges, _ = bstat(np.abs(nw_residual))
        redchisq = np.nanmean(np.abs(nw_residual))
        ax1.scatter(
            bin_edges[:-1], mean, label="Noise-weighted residual", s=2,
        )
        ax1.set_xlabel("Baseline length [m]")
        ax0.set_ylabel("Abs(visibilities) [Jy]")
        ax1.set_ylabel("Noise-weighted residual [1]")
        ax1.set_yscale("log")
        ax1.axhline(1, color="gray", linestyle="dashed", label="Expected value")
        ax1.axhline(
            redchisq,
            color="red",
            linestyle="dotted",
            label=f"Reduced chi²: {redchisq:.1f}",
        )
        ax1.set_ylim([0.1, None])
        ax0.legend()
        ax1.legend()
        plt.savefig(path("data_histogram.png"))
        plt.close()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
