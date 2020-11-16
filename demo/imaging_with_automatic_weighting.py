# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse

import numpy as np

import nifty7 as ift
import resolve as rve
from os.path import isfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=int, default=1)
    parser.add_argument("--use-cached", action="store_true")
    parser.add_argument("ms")
    args = parser.parse_args()

    rve.set_nthreads(args.j)
    rve.set_wgridding(False)
    obs = rve.ms2observations(args.ms, "DATA", False, 0, "stokesiavg")[0]

    rve.set_epsilon(1 / 10 / obs.max_snr())
    fov = np.array([3, 1.5]) * rve.ARCMIN2RAD
    npix = np.array([4096, 2048])
    npix = np.array([4096, 2048]) / 4  # FIXME QUICK
    dom = ift.RGSpace(npix, fov / npix)
    logsky = ift.SimpleCorrelatedField(
        dom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5)
    )
    diffuse = logsky.exp()
    inserter = rve.PointInserter(dom, np.array([[0, 0], [0.7, -0.34]]) * rve.AS2RAD)
    points = ift.InverseGammaOperator(
        inserter.domain, alpha=0.5, q=0.2 / dom.scalar_dvol
    ).ducktape("points")
    points = inserter @ points
    sky = points + diffuse
    npix = 2500
    effuv = np.linalg.norm(obs.effective_uv().T, axis=1)
    assert obs.nfreq == obs.npol == 1
    dom = ift.RGSpace(npix, 2 * np.max(effuv) / npix)
    logwgt = ift.SimpleCorrelatedField(
        dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), "invcov"
    )
    li = ift.LinearInterpolator(dom, effuv)
    weightop = ift.makeOp(obs.weight) @ (
        rve.AddEmptyDimension(li.target) @ li @ logwgt.exp()
    ) ** (-2)

    plotter = rve.Plotter("png", "plots")
    plotter.add("logsky", logsky)
    plotter.add("power spectrum logsky", logsky.power_spectrum)
    plotter.add("bayesian weighting", logwgt.exp())
    plotter.add("power spectrum bayesian weighting", logwgt.power_spectrum)

    # MAP points with original weights
    lh = rve.ImagingLikelihood(obs, points)
    ham = ift.StandardHamiltonian(lh)
    state = rve.MinimizationState(0.1 * ift.from_random(ham.domain), [])
    mini = ift.NewtonCG(ift.GradientNormController(name="newton", iteration_limit=4))
    if args.use_cached and isfile("stage0"):
        state = rve.MinimizationState.load("stage0")
    else:
        state = rve.simple_minimize(ham, state.mean, 0, mini)
        plotter.plot("stage0", state)
        state.save("stage0")

    # MAP diffuse with original weights
    lh = rve.ImagingLikelihood(obs, sky)
    plotter.add_histogram(
        "normalized residuals (original weights)", lh.normalized_residual
    )
    ham = ift.StandardHamiltonian(lh)
    state = rve.MinimizationState(
        ift.MultiField.union([0.1 * ift.from_random(diffuse.domain), state.mean]), []
    )
    mini = ift.NewtonCG(ift.GradientNormController(name="newton", iteration_limit=20))
    if args.use_cached and isfile("stage1"):
        state = rve.MinimizationState.load("stage1")
    else:
        state = rve.simple_minimize(ham, state.mean, 0, mini)
        plotter.plot("stage1", state)
        state.save("stage1")

    # MGVI weights
    lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, weightop)
    plotter.add_histogram(
        "normalized residuals (learned weights)", lh.normalized_residual
    )
    ic = ift.AbsDeltaEnergyController(0.1, 3, 100, name="Sampling")
    ham = ift.StandardHamiltonian(lh, ic)
    cst = sky.domain.keys()
    state = rve.MinimizationState(
        ift.MultiField.union([0.1 * ift.from_random(weightop.domain), state.mean]), []
    )
    mini = ift.VL_BFGS(ift.GradientNormController(name="bfgs", iteration_limit=20))
    if args.use_cached and isfile("stage2"):
        state = rve.MinimizationState.load("stage2")
    else:
        for ii in range(10):
            state = rve.simple_minimize(ham, state.mean, 0, mini, cst, cst)
            plotter.plot(f"stage2_{ii}", state)
        state.save("stage2")

    # Reset sky
    state = rve.MinimizationState(
        ift.MultiField.union([state.mean, 0.1 * ift.from_random(sky.domain)]), []
    )

    # MGVI sky
    ic = ift.AbsDeltaEnergyController(0.1, 3, 200, name="Sampling")
    ham = ift.StandardHamiltonian(lh, ic)
    cst = list(points.domain.keys()) + list(weightop.domain.keys())
    mini = ift.NewtonCG(ift.GradientNormController(name="newton", iteration_limit=15))
    for ii in range(4):
        fname = f"stage3_{ii}"
        if args.use_cached and isfile(fname):
            state = rve.MinimizationState.load(fname)
        else:
            state = rve.simple_minimize(ham, state.mean, 5, mini, cst, cst)
            plotter.plot(f"stage3_{ii}", state)
            state.save(fname)

    # Sky + weighting simultaneously
    ic = ift.AbsDeltaEnergyController(0.1, 3, 200, name="Sampling")
    ham = ift.StandardHamiltonian(lh, ic)
    mini = ift.NewtonCG(ift.GradientNormController(name="newton", iteration_limit=15))
    for ii in range(30):
        fname = f"stage4_{ii}"
        if args.use_cached and isfile(fname):
            state = rve.MinimizationState.load(fname)
        else:
            state = rve.simple_minimize(ham, state.mean, 5, mini)
            plotter.plot(f"stage4_{ii}", state)
            state.save(fname)


if __name__ == "__main__":
    main()
