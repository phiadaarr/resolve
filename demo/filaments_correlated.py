# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift
import resolve as rve
import matplotlib.pyplot as plt


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

    ### 1.Generate c0 and phi0 by correlated field model

    # random seed
    ift.random.push_sseq_from_seed(68)


    # c0 by correlated field model, initial density (rho0) = exp(c0)
    cfmaker_c0 = ift.CorrelatedFieldMaker('c0')
    # add fluctuations, flexibility, asperity, loglogavgslope
    cfmaker_c0.add_fluctuations(normalized_domain, (4.0, 1.0), (1.2, 0.4), None, (-4., 1.0), 'c0')
    cfmaker_c0.set_amplitude_total_offset(21., (1.0, 0.1))
    Correlated_field_c0 = cfmaker_c0.finalize()
    C0 = Correlated_field_c0


    # phi0 by correlated field model
    cfmaker_phi0 = ift.CorrelatedFieldMaker('phi0')
    # add fluctuations, flexibility, asperity, loglogavgslope
    cfmaker_phi0.add_fluctuations(normalized_domain, (0.5, 0.25), (0.6, 0.2), None, (-5., 1.0), 'phi0')
    cfmaker_phi0.set_amplitude_total_offset(0., (1.0, 0.1))
    Correlated_field_phi0 = cfmaker_phi0.finalize()
    # minus lognormal field phi0
    Phi0_ = Correlated_field_phi0
    Phi0 = -1 * ift.exp(Phi0_)

    ### 2.Calculate initial wave function operator Psi_0

    hbar = 5 * 10 ** -3
    a = 0.05 # time scale
    Half_operator_ = ift.ScalingOperator(C0.target, 0.5)
    Hbar_operator = ift.ScalingOperator(Phi0.target, -1j / hbar)
    Complexifier = ift.Realizer(Phi0.target).adjoint
    Phase_operator = Hbar_operator @ Complexifier

    Half_operator = Half_operator_ @ Complexifier
    Psi_0 = ift.exp(Half_operator(C0) + Phase_operator(Phi0))
    # psi_0 = np.exp(c0 / 2 - 1j * phi0 / hbar)
    Psi_0h = fft(Psi_0)

    ### 3.Initial filament density operator Rho0
    Rho0 = ift.exp(C0)
    # Rho0 = ift.exp(Two_times_operator(ift.log(Psi_0)).real)

    ### 4.Wave function_1 at time a(by free propagator in QM)

    # length of k vector for each pixel
    k_values = harmonic_space.get_k_length_array()
    propagator_h = ift.exp(-1j * hbar * a / 2 * (k_values) ** 2)

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
    rve.set_wgridding(False)

    # Load data
    obs = rve.Observation.load("2052-2MHz_with_learned_weights.npz")
    rve.set_epsilon(1e-6)
    rve.set_nthreads(4)


    xfov = yfov = "250as"
    #npix = 4000
    npix = 3000
    #npix = 30

    fov = np.array([rve.str2rad(xfov), rve.str2rad(yfov)])

    npix = np.array([npix, npix])
    dom = ift.RGSpace(npix, fov / npix)

    inserter = rve.PointInserter(dom, [[0, 0]])
    points = ift.InverseGammaOperator(
        inserter.domain, alpha=0.5, q=0.2 / dom.scalar_dvol
    ).ducktape("points")
    points = inserter @ points
    #filaments = get_filament_prior(dom).scale(1e10) # 1e10
    filaments = get_filament_prior(dom)
    sky = filaments + points

    '''
    ### load the state
    state = rve.MinimizationState.load("filaments6")
    ift.single_plot(sky(state.mean))
    exit()
    '''

    p = ift.Plot()
    for ii in range(9):
        p.add(sky.log10()(ift.from_random(sky.domain)))
    p.output(name="prior_samples.png")

    # npix = 2500
    # effuv = np.linalg.norm(obs.effective_uv().T, axis=1)
    # assert obs.nfreq == obs.npol == 1
    # dom = ift.RGSpace(npix, 2 * np.max(effuv) / npix)
    # logwgt = ift.SimpleCorrelatedField(
    #     dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), "invcov"
    # )
    # li = ift.LinearInterpolator(dom, effuv)
    # weightop = ift.makeOp(obs.weight) @ (
    #     rve.AddEmptyDimension(li.target) @ li @ logwgt.exp()
    # ) ** (-2)


    mini = ift.NewtonCG(ift.GradientNormController(name="newton", iteration_limit=5))
    # Fit point source only
    state = rve.MinimizationState(0.1 * ift.from_random(sky.domain), [])
    lh = rve.ImagingLikelihood(obs, sky)
    ham = ift.StandardHamiltonian(
        lh, ift.AbsDeltaEnergyController(0.5, iteration_limit=100)
    )
    cst = filaments.domain.keys()
    state = rve.simple_minimize(
        ham, state.mean, 0, mini, constants=cst, point_estimates=cst
    )


    # Fit diffuse + points
    for ii in range(20):
        state = rve.simple_minimize(ham, state.mean, 0, mini)
        if ii >= 19:
            state.save(f"filaments{ii}")

        #ift.single_plot(sky.log10()(state.mean), name=f"sky{ii}.png")

        plot = ift.Plot()
        plot.add(sky.log10()(state.mean), title="sky{:02d}(logscale) a=0.05".format(ii), fontsize=20)
        plot.output(ny=1, nx=1, xsize=20, ysize=20, name="sky{:02d}_logscale_weights.png".format(ii))

        R = rve.StokesIResponse(obs, dom)
        print((R @ sky)(state.mean))
        print(obs.vis)
        ift.extra.minisanity(
            obs.vis, lambda pos: ift.makeOp(obs.weight), R @ sky, state.mean
        )


if __name__ == "__main__":
    main()
