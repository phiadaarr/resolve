# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse
from functools import reduce
from operator import add
from os.path import isfile, splitext

import numpy as np

import nifty7 as ift
import resolve as rve


def main():
    parser = argparse.ArgumentParser()
    # TODO Add modes: reconstruct, plot
    # TODO OU process for automatic weighting
    parser.add_argument("-j", type=int, default=1)
    parser.add_argument("--use-cached", action="store_true")
    parser.add_argument("--use-wgridding", action="store_true")
    parser.add_argument(
        "--data-column",
        default="DATA",
        help="Only active if a measurement set is read.",
    )
    parser.add_argument("--point", action="append", nargs=2)
    parser.add_argument("ms", type=str)
    parser.add_argument("xfov", type=str)
    parser.add_argument("yfov", type=str)
    parser.add_argument("xpix", type=int)
    parser.add_argument("ypix", type=int)
    parser.add_argument("diffusefluxlevel", type=float)
    args = parser.parse_args()

    ############################################################################
    # Define likelihood(s)
    ############################################################################
    if splitext(args.ms)[1] == ".npz":
        obs = rve.Observation.load(args.ms)
    else:
        obs = rve.ms2observations(args.ms, args.data_column, False, 0, "stokesiavg")[0]

    rve.set_nthreads(args.j)
    rve.set_wgridding(args.use_wgridding)
    fov = np.array([rve.str2rad(args.xfov), rve.str2rad(args.yfov)])
    npix = np.array([args.xpix, args.ypix])
    rve.set_epsilon(1 / 10 / obs.max_snr())
    plotter = rve.Plotter("png", "plots")

    polmode = obs.polarization.has_crosshanded()

    # TODO Add mode with independent noise learning
    effuv = np.linalg.norm(obs.effective_uv().T, axis=1)
    dom = ift.RGSpace((npix_wgts := 2500), 2 * np.max(effuv) / npix_wgts)
    if not polmode:
        assert obs.nfreq == obs.npol == 1
        logwgt = ift.SimpleCorrelatedField(
            dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), "invcov"
        )
        li = ift.LinearInterpolator(dom, effuv)
        weightop = ift.makeOp(obs.weight) @ (
            rve.AddEmptyDimension(li.target) @ li @ logwgt.exp()
        ) ** (-2)
    else:
        assert obs.nfreq == 1
        cfm = ift.CorrelatedFieldMaker.make(0, (2, 2), "invcov", 4)
        cfm.add_fluctuations(dom, (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5))
        logwgt = cfm.finalize(0)
        li = rve.LinearOperatorOverAxis(ift.LinearInterpolator(dom, effuv), logwgt.target)
        empty = rve.AddEmptyDimensionAtEnd(li.target)
        ift.extra.check_linear_operator(empty)
        ift.extra.check_linear_operator(li)
        weightop = ift.makeOp(obs.weight) @ (empty @ li @ logwgt.exp()) ** (-2)
        plotter.add("bayesian weighting", logwgt.exp())
        plotter.add("power spectrum bayesian weighting", cfm.power_spectrum)

    dom = ift.RGSpace(npix, fov / npix)
    if polmode:
        assert args.point is None
        print("Instantiate polarization model")
        params = {
            "i": args.diffusefluxlevel,
            "q": 0,
            "u": 0,
        }
        opdct = {}
        for kk in ["i", "q", "u"]:
            opdct[kk] = ift.SimpleCorrelatedField(
                dom,
                args.diffusefluxlevel,
                (1, 0.1),
                (5, 1),
                (1.2, 0.4),
                (0.2, 0.2),
                (-2, 0.5),
                prefix=f"log{kk}",
            )
        logop = reduce(add, [vv.ducktape_left(kk) for kk, vv in opdct.items()])
        mexp = rve.polarization_matrix_exponential(logop.target)
        # ift.extra.check_operator(mexp, ift.from_random(mexp.domain)*0.1, ntries=5)
        sky = mexp @ logop
        duckI = ift.ducktape(None, sky.target, "I")
        duckQ = ift.ducktape(None, sky.target, "Q")
        duckU = ift.ducktape(None, sky.target, "U")
        polarized_part = duckQ(sky) ** 2 + duckU(sky) ** 2  # + V ** 2
        frac_pol = polarized_part * (polarized_part + duckI(sky) ** 2).reciprocal()
        plotter.add("logstokesi", duckI(sky).log())
        plotter.add("stokesq", duckQ(sky))
        plotter.add("stokesu", duckU(sky))
        plotter.add("fractional_polarization", frac_pol, vmin=0, vmax=1, cmap="Greys")
    else:
        logsky = ift.SimpleCorrelatedField(
            dom,
            args.diffusefluxlevel,
            (1, 0.1),
            (5, 1),
            (1.2, 0.4),
            (0.2, 0.2),
            (-2, 0.5),
        )
        diffuse = logsky.exp()
        if args.point is not None:
            ppos = []
            for point in args.point:
                ppos.append([rve.str2rad(point[0]), rve.str2rad(point[1])])
            inserter = rve.PointInserter(dom, ppos)
            points = ift.InverseGammaOperator(
                inserter.domain, alpha=0.5, q=0.2 / dom.scalar_dvol
            ).ducktape("points")
            points = inserter @ points
            sky = diffuse + points
        else:
            sky = diffuse
        plotter.add("logsky", logsky)
        plotter.add("power spectrum logsky", logsky.power_spectrum)
        plotter.add("bayesian weighting", logwgt.exp())
        plotter.add("power spectrum bayesian weighting", logwgt.power_spectrum)

    ############################################################################
    # MINIMIZATION
    ############################################################################
    if rve.mpi.master:
        if args.point is not None:
            # MAP points with original weights
            lh = rve.ImagingLikelihood(obs, points)
            ham = ift.StandardHamiltonian(lh)
            state = rve.MinimizationState(0.1 * ift.from_random(ham.domain), [])
            mini = ift.NewtonCG(
                ift.GradientNormController(name="newton", iteration_limit=4)
            )
            if args.use_cached and isfile("stage0"):
                state = rve.MinimizationState.load("stage0")
            else:
                state = rve.simple_minimize(ham, state.mean, 0, mini)
                plotter.plot("stage0", state)
                state.save("stage0")

        # MAP diffuse with original weights
        lh = rve.ImagingLikelihood(obs, sky, polmode)
        plotter.add_histogram(
            "normalized residuals (original weights)", lh.normalized_residual
        )
        ham = ift.StandardHamiltonian(lh)
        if polmode:
            fld = 0.1 * ift.from_random(sky.domain)
        else:
            if args.point is None:
                fld = 0.1 * ift.from_random(diffuse.domain)
            else:
                fld = ift.MultiField.union(
                    [0.1 * ift.from_random(diffuse.domain), state.mean]
                )
        state = rve.MinimizationState(fld, [])
        mini = ift.NewtonCG(
            ift.GradientNormController(name="newton", iteration_limit=40)
        )
        if args.use_cached and isfile("stage1"):
            state = rve.MinimizationState.load("stage1")
        else:
            state = rve.simple_minimize(ham, state.mean, 0, mini)
            plotter.plot("stage1", state)
            state.save("stage1")

        # Only weights
        lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, weightop, polmode)
        plotter.add_histogram(
            "normalized residuals (learned weights)", lh.normalized_residual
        )
        ic = ift.AbsDeltaEnergyController(0.1, 3, 100, name="Sampling")
        ham = ift.StandardHamiltonian(lh, ic)
        cst = sky.domain.keys()
        state = rve.MinimizationState(
            ift.MultiField.union([0.1 * ift.from_random(weightop.domain), state.mean]),
            [],
        )
        mini = ift.VL_BFGS(ift.GradientNormController(name="bfgs", iteration_limit=20))
        if args.use_cached and isfile("stage2"):
            state = rve.MinimizationState.load("stage2")
        else:
            for ii in range(10):
                state = rve.simple_minimize(ham, state.mean, 0, mini, cst, cst)
                plotter.plot(f"stage2_{ii}", state)
            state.save("stage2")

    if rve.mpi.mpi:
        if not rve.mpi.master:
            state = None
        state = rve.mpi.comm.bcast(state, root=0)
    ift.random.push_sseq_from_seed(42)

    # MGVI sky
    ic = ift.AbsDeltaEnergyController(0.1, 3, 200, name="Sampling")
    lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, weightop, polmode)
    ham = ift.StandardHamiltonian(lh, ic)
    if args.point is not None:
        cst = list(points.domain.keys()) + list(weightop.domain.keys())
    else:
        cst = list(weightop.domain.keys())
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
    ic = ift.AbsDeltaEnergyController(0.1, 3, 700, name="Sampling")
    ham = ift.StandardHamiltonian(lh, ic)
    for ii in range(30):
        if ii < 5:
            mini = ift.VL_BFGS(
                ift.GradientNormController(name="newton", iteration_limit=15)
            )
        else:
            mini = ift.NewtonCG(
                ift.GradientNormController(name="newton", iteration_limit=15)
            )
        fname = f"stage4_{ii}"
        if args.use_cached and isfile(fname):
            state = rve.MinimizationState.load(fname)
        else:
            state = rve.simple_minimize(ham, state.mean, 5, mini)
            plotter.plot(f"stage4_{ii}", state)
            state.save(fname)


if __name__ == "__main__":
    main()
