# SPDX-License-Identifier: GPL-4.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import argparse
import os
from os.path import join
from socket import gethostname
from time import time

import numpy as np

import nifty7 as ift
import resolve as rve


def sf_sky_model(dom, with_points=True, with_diffuse=True):
    i0 = ift.SimpleCorrelatedField(
        dom,
        11,
        (1, 0.1),
        (5, 1),
        (1.2, 0.4),
        (0.2, 0.2),
        (-2, 0.5),
        prefix="diffuse_i0",
    )
    loginvgamma = ift.LogInverseGammaOperator(dom, 1, 1e-5 / dom.scalar_dvol).ducktape(
        "points_i0"
    )
    diffuse = i0.exp()
    points = loginvgamma.exp()
    if with_points:
        if with_diffuse:
            return diffuse + points
        return points
    if with_diffuse:
        return diffuse
    raise NotImplementedError


rve.set_wgridding(True)
lo_hi_index = 0, 8
if gethostname() == "valparaiso":
    lo_hi_index = 0, 1
obs = rve.Observation.load("eso137.npz", lo_hi_index).average_stokesi()

# rve.set_epsilon(1 / 10 / obs.max_snr()) FIXME Compute with MPI
rve.set_epsilon(0.0005)
fov = np.array([2, 2]) * rve.DEG2RAD
npix = np.array([4000, 4000])
dom = ift.RGSpace(npix, fov / npix)
with_diffuse, with_points = False, True
sky = sf_sky_model(dom, with_points, with_diffuse)


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

    lh = rve.ImagingLikelihood(obs, sky)

    ham = ift.StandardHamiltonian(
        lh, ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=100)
    )
    initial = 0.1 * ift.from_random(lh.domain)
    state = rve.MinimizationState(initial)

    # ift.single_plot(sky(initial), name=f"initial.png")
    # R = rve.StokesIResponse(obs, dom)
    # print((R@sky)(initial))
    # print(obs.vis)
    # exit()

    print("Sky timing")
    ift.exec_time(sky)
    print("Likelihood timing")
    ift.exec_time(lh)
    for ii in range(50):
        state = rve.simple_minimize(
            ham,
            state.mean,
            0,
            ift.NewtonCG(
                ift.AbsDeltaEnergyController(0.5, iteration_limit=5, name="Newton")
            ),
        )
        state.save(path("minimization_state"))

        with open(path(f"likelihood.csv"), "a") as f:
            # Iteration index, time, energy, redchi^2, frequency min, max, mean, effective_frequency
            s = [ii, time()]
            e = lh(state.mean).val
            s += [e, e / obs.n_data_effective]
            s = [str(ss) for ss in s]
            f.write(",".join(s) + "\n")
        if rve.mpi.master:
            ift.single_plot(sky(state.mean), name=path(f"debug_{ii}.png"))
            rve.field2fits(
                sky(state.mean), path(f"debug_{ii}.fits"), True, obs.direction
            )
            for kk, vv in state.mean.items():
                if (
                    len(vv.shape) != 2
                    or len(vv.domain) != 1
                    or not isinstance(vv.domain[0], ift.RGSpace)
                ):
                    continue
                ift.single_plot(vv, name=path(f"latent_{kk}_{ii}.png"))
                rve.field2fits(vv, path(f"latent_{kk}_{ii}.fits"), True, obs.direction)


if __name__ == "__main__":
    main()
