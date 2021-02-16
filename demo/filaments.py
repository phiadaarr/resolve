# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift
import resolve as rve


def main():
    # NOTE You can use MPI with this script. Start it with `mpirun -np X python3
    # ...py`. Then it parallizes over the MGVI samples. This scales very well.

    # NOTE Here you can turn on the parallelization of FFT and the radio response.
    # rve.set_nthreads(4)

    # NOTE This enables so-called wgridding. This is a wide-field effect. See
    # Equation (1.30) in
    # https://wwwmpa.mpa-garching.mpg.de/~parras/papers/dissertation.pdf.
    # w-gridding becomes important if w*(n-1) is not negligable. It could be
    # relevant for this data set but probably not.
    # rve.set_wgridding(True)

    # Load data
    obs = rve.Observation.load("CYG-ALL-13360-8MHZfield0.npz")
    rve.set_epsilon(1 / 10 / obs.max_snr())

    xfov = yfov = "150as", "150as"
    npix = 4000
    npix = 400
    fov = np.array([rve.str2rad(xfov), rve.str2rad(yfov)])
    npix = np.array([npix, npix])
    dom = ift.RGSpace(npix, fov / npix)

    # TODO Define sky operator
    raise NotImplementedError
    # sky =
    #
    # NOTE In the central region of the galaxy there are two very bright point
    # sources. Probably the filament prior will have problems with them. Let's
    # see.

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
    plotter.add("bayesian weighting", logwgt.exp())
    plotter.add("power spectrum bayesian weighting", logwgt.power_spectrum)

    ############################################################################
    # MINIMIZATION
    ############################################################################
    if rve.mpi.master:
        # MAP diffuse with original weights
        lh = rve.ImagingLikelihood(obs, sky)
        plotter.add_histogram(
            "normalized residuals (original weights)", lh.normalized_residual
        )
        ham = ift.StandardHamiltonian(lh)
        fld = 0.1 * ift.from_random(sky.domain)

        state = rve.MinimizationState(fld, [])
        mini = ift.NewtonCG(
            ift.GradientNormController(name="newton", iteration_limit=20)
        )
        # state = rve.MinimizationState.load("stage1")
        state = rve.simple_minimize(ham, state.mean, 0, mini)
        plotter.plot("stage1", state)
        state.save("stage1")

        # Only weights. This learns what is plotted in Figure 8 of
        # https://arxiv.org/pdf/2008.11435.pdf
        lh = rve.ImagingLikelihood(obs, sky, inverse_covariance_operator=weightop)
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
        # state = rve.MinimizationState.load("stage2")
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
    lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, weightop)
    ham = ift.StandardHamiltonian(lh, ic)
    cst = list(weightop.domain.keys())
    mini = ift.NewtonCG(ift.GradientNormController(name="newton", iteration_limit=15))
    for ii in range(4):
        fname = f"stage3_{ii}"
        # state = rve.MinimizationState.load(fname)
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
        # state = rve.MinimizationState.load(fname)
        state = rve.simple_minimize(ham, state.mean, 5, mini)
        plotter.plot(f"stage4_{ii}", state)
        state.save(fname)


if __name__ == "__main__":
    main()
