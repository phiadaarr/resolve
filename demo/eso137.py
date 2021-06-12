# SPDX-License-Identifier: GPL-4.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import argparse
import os
from functools import reduce
from operator import add
from os.path import join
from socket import gethostname
from time import time

import numpy as np

import nifty7 as ift
import resolve as rve


def mf_sky_model(dom, with_points=True, with_diffuse=True):
    i0 = ift.SimpleCorrelatedField(
        dom,
        10,
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
    alpha_diffuse = ift.SimpleCorrelatedField(
        dom,
        -1,
        (1, 0.1),
        (1, 1),
        (1.2, 0.4),
        (0.2, 0.2),
        (-2, 0.5),
        prefix=f"diffuse_alpha",
    )  # .softplus().scale(-1)
    # alpha_points = (
    #     (ift.Adder(ift.full(dom, 2)) @ ift.ducktape(dom, None, "points_alpha"))
    #     .softplus()
    #     .scale(-1)
    # )
    alpha_points = ift.ducktape(dom, None, "points_alpha")
    diffuse = i0.ducktape_left("logi0") + alpha_diffuse.ducktape_left("diffuse_alpha")
    points = loginvgamma.ducktape_left("logpoints") + alpha_points.ducktape_left(
        "points_alpha"
    )
    if with_points:
        if with_diffuse:
            return diffuse + points
        return points
    if with_diffuse:
        return diffuse
    raise NotImplementedError


def inverse_covariance_operator(observation):
    raise NotImplementedError
    npix = 2500
    effuv = np.linalg.norm(observation.effective_uv().T, axis=1)
    assert observation.nfreq == observation.npol == 1
    dom = ift.RGSpace(npix, 2 * np.max(effuv) / npix)
    logwgt = ift.SimpleCorrelatedField(
        dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), "invcov"
    )
    li = ift.LinearInterpolator(dom, effuv)
    return ift.makeOp(observation.weight) @ (
        rve.AddEmptyDimension(li.target) @ li @ logwgt.exp()
    ) ** (-2)


rve.set_wgridding(True)
n_imaging_bands = 10
if gethostname() == "valparaiso":
    n_imaging_bands = 2
obs_list, global_nu0 = rve.Observation.load_mf(
    "eso137.npz", n_imaging_bands, rve.mpi.comm
)
obs_list = [obs.average_stokesi() for obs in obs_list]

# rve.set_epsilon(1 / 10 / obs.max_snr()) FIXME Compute with MPI
rve.set_epsilon(0.0005)
fov = np.array([2, 2]) * rve.DEG2RAD
npix = np.array([4000, 4000])
dom = ift.RGSpace(npix, fov / npix)
with_diffuse, with_points = True, False
sky = mf_sky_model(dom, with_points, with_diffuse)

print(f"Data freqs ({rve.mpi.rank}):", [oo.freq for oo in obs_list])


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

    local_lhs = []
    local_a0s = []
    local_skys = []
    for oo in obs_list:
        a0 = np.log(oo.freq.mean() / global_nu0)
        sky_op = []
        fa = lambda s: ift.FieldAdapter(dom, s)
        if with_diffuse:
            sky_op.append((fa("logi0") + fa("diffuse_alpha").scale(a0)).exp())
        if with_points:
            sky_op.append((fa("logpoints") + fa("points_alpha").scale(a0)).exp())
        ss = reduce(add, sky_op)
        local_skys.append(ss)
        local_lhs.append(rve.ImagingLikelihood(oo, ss))
        local_a0s.append(a0)
        print(f"Shape visibilities (rank {rve.mpi.rank})", oo.vis.shape)
    lh = rve.AllreduceSum(local_lhs, rve.mpi.comm) @ sky

    ham = ift.StandardHamiltonian(
        lh,
        ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=1000, name="CG"),
    )
    initial = 0.1 * ift.from_random(lh.domain)
    for ii, ss in enumerate(local_skys):
        ift.single_plot((ss @ sky)(initial), name=f"initial{ii}.png")
    state = rve.MinimizationState(initial)

    # print("Sky timing")
    # ift.exec_time(sky)
    # print("Likelihood timing")
    # ift.exec_time(lh)
    for ii in range(50):
        # FIXME Assert that random state is synchronized
        mini = rve.Minimization(ham, state.mean, 2, [], [], None)
        state = mini.minimize(
            ift.NewtonCG(
                ift.AbsDeltaEnergyController(0.5, iteration_limit=5, name="Newton")
            )
        )
        state.save(path("minimization_state"))

        if len(state) == 0:
            meansky = sky.force(state.mean)
        else:
            sc = ift.StatCalculator()
            for ss in state:
                sc.add(sky.force(ss))
            meansky = sc.mean
        with open(path(f"likelihood_{rve.mpi.rank}.csv"), "a") as f:
            # Iteration index, time, energy, redchi^2, frequency min, max, mean, effective_frequency
            s = [ii, time()]
            for jj, (oo, ll, aa) in enumerate(zip(obs_list, local_lhs, local_a0s)):
                e = local_lhs[jj](meansky).val  # FIXME for MGVI
                s += [e, e / oo.n_data_effective]
                s += [oo.freq.min(), oo.freq.max(), oo.freq.mean()]
                s += [aa]
            s = [str(ss) for ss in s]
            f.write(",".join(s) + "\n")
        if rve.mpi.master:
            for kk, vv in meansky.items():
                ift.single_plot(vv, name=path(f"debug_{kk}_{ii}.png"))
                rve.field2fits(
                    vv, path(f"debug_{kk}_{ii}.fits"), True, obs_list[0].direction
                )
            for kk, vv in state.mean.items():
                if (
                    len(vv.shape) != 2
                    or len(vv.domain) != 1
                    or not isinstance(vv.domain[0], ift.RGSpace)
                ):
                    continue
                ift.single_plot(vv, name=path(f"latent_{kk}_{ii}.png"))
                rve.field2fits(
                    vv, path(f"latent_{kk}_{ii}.fits"), True, obs_list[0].direction
                )


if __name__ == "__main__":
    main()
