# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift
import resolve as rve


class DomainChanger(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        return ift.makeField(self._tgt(mode), x.val)


def get_filament_prior(domain):
    normalized_domain = ift.RGSpace(domain.shape)
    ### Define harmonic space and harmonic transform
    harmonic_space = normalized_domain.get_default_codomain()
    # HarmonicTransformOperator is actually Hartley transform, only take real value
    HT = ift.HarmonicTransformOperator(harmonic_space, normalized_domain)
    # FFTOperator(real FFT) produce complex number, we need to take real number(by .real)
    ifft = ift.FFTOperator(harmonic_space, normalized_domain)  # from k-space to position space
    fft = ifft.inverse

    ### 1.Generate gaussian random field c0 and phi0

    # c0 = gaussian random field, initial density (rho0) = exp(c0)
    # amplitude spectrum of c0(P ~ Amp^2 ~ k^-2)
    def amp_spec_c0(k):
        P0, k0, gamma = [0.00001, 1, 2]
        return np.sqrt(P0 / ((1.0 + (k / k0) ** 2) ** (gamma / 2)))

    # Operators(for Operators we use upper-case)
    C0h = ift.create_power_operator(
        harmonic_space, power_spectrum=amp_spec_c0
    ).ducktape("c0h")
    C0 = HT(C0h)

    # phi0 = gaussian random field
    # amplitude spectrum(P~ Amp^2 ~ k^-6)
    def amp_spec(k):
        P0, k0, gamma = [0.01, 1, 6]
        return np.sqrt(P0 / ((1.0 + (k / k0) ** 2) ** (gamma / 2)))

    # Operators
    Phi0h = ift.create_power_operator(harmonic_space, power_spectrum=amp_spec).ducktape(
        "phi0h"
    )
    Phi0 = HT(Phi0h)

    ### 2.Calculate initial wave function operator Psi_0

    hbar = 5 * 10 ** -3
    a = 1.0  # time scale
    Half_operator_ = ift.ScalingOperator(C0.target, 0.5)
    Hbar_operator = ift.ScalingOperator(Phi0.target, -1j / hbar)
    Complexifier = ift.Realizer(Phi0.target).adjoint
    Phase_operator = Hbar_operator @ Complexifier

    Half_operator = Half_operator_ @ Complexifier
    Psi_0 = ift.exp(Half_operator(C0) - Phase_operator(Phi0))
    # psi_0 = np.exp(c0 / 2 - 1j * phi0 / hbar)
    Psi_0h = fft(Psi_0)

    ### 3.Initial filament density operator Rho0
    Rho0 = ift.exp(C0)
    # Rho0 = ift.exp(Two_times_operator(ift.log(Psi_0)).real)

    ### 4.Wave function_1 at time a(by free propagator in QM)

    # length of k vector for each pixel
    k_values = harmonic_space.get_k_length_array()

    propagator_h = ift.exp(
        -1j * hbar * a / 2 * (k_values) ** 2
    )  # to put noise, use k_values_n instead of k_values

    # propagator operator in harmonic space
    Propagator_h = ift.makeOp(propagator_h)

    # Wave function_1 operator(Psi_1)
    Psi_1h = Propagator_h(Psi_0h)  # psi_1h = propagator_h * psi_0h
    Psi_1 = ifft(Psi_1h)

    ### 5.Calculate filament density operator(Rho1)

    # absolute square operator for complex number
    conjOP = ift.ConjugationOperator(Psi_1._target)
    rls_psf = ift.Realizer(Psi_1.target)

    # filament density operator Rho1 at time a
    Rho1 = rls_psf @ (ift.ScalingOperator(Psi_1.target, 1) * conjOP) @ Psi_1
    ### Rho1 is the prior (rho1 = Rho1(xi))
    return DomainChanger(Rho1.target, domain) @ Rho1


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

    xfov = yfov = "150as"
    npix = 4000
    npix = 400

    fov = np.array([rve.str2rad(xfov), rve.str2rad(yfov)])

    npix = np.array([npix, npix])
    dom = ift.RGSpace(npix, fov / npix)

    sky = get_filament_prior(dom)

    p = ift.Plot()
    for _ in range(9):
        p.add(sky(ift.from_random(sky.domain)))
    p.output(name="prior_samples.png")

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
