# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse

import matplotlib.pyplot as plt
import numpy as np

import nifty7 as ift
import resolve as rve

# FIXME Add damping to Wiener integration
# FIXME cumsum over first dimensions may be performance-wise suboptimal


def mf_logsky(domain, freq, prefix, plotter):
    assert np.all(np.diff(freq) > 0)  # need ordered frequencies
    nfreq = len(freq)
    freqdom = rve.IRGSpace(freq)
    assert freqdom.size == nfreq

    # FIXME Figure out why the values are so freaking big/small
    flexibility, asperity = (1e-11, 1e-14), (1e14, 1e14)

    a0 = ift.SimpleCorrelatedField(domain, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5),
                                   prefix=f'{prefix}a0')
    b0 = ift.SimpleCorrelatedField(domain, 0, (1e-7, 1e-7), (1e-7, 1e-7), (1.2, 0.4), (0.2, 0.2), (-2, 0.5),
                                   prefix=f'{prefix}b0')
    # FIXME Is there a bug in the b0 handling?
    b0 = ift.ScalingOperator(domain, 0.).ducktape(f'{prefix}b0')
    plotter.add("a0", a0)
    plotter.add("b0", b0)

    # IDEA Try to use individual power spectra
    # FIXME Support fixed variance for zero mode
    cfm = ift.CorrelatedFieldMaker.make(0., (1, 0.00001), f'{prefix}freqxi', total_N=2*(nfreq-1))
    # FIXME Support fixed fluctuations
    cfm.add_fluctuations(domain, (1, 0.00001), (1.2, 0.4), (0.2, 0.2), (-2, 0.5))
    freqxi = cfm.finalize(0)

    # FIXME Make sure that it is standard normal distributed in uncorrelated directions
    # fld = freqxi(ift.from_random(freqxi.domain)).val
    # mean = np.mean(fld, axis=(2, 3))
    # std = np.std(fld, axis=(2, 3))
    # print(np.mean(mean), np.std(mean))
    # print(np.mean(std), np.std(std))

    intop = rve.WienerIntegrations(freqdom, domain)
    freqxi = rve.Reshaper(freqxi.target, intop.domain) @ freqxi
    expander = ift.ContractionOperator(intop.domain, (0, 1)).adjoint
    vol = freqdom.dvol

    flex = ift.LognormalTransform(*flexibility, prefix + 'flexibility', 0)
    dom = intop.domain[0]
    vflex = np.empty(dom.shape)
    vflex[0] = vflex[1] = np.sqrt(vol)
    vflex = ift.DiagonalOperator(ift.makeField(dom, vflex),
                                 domain=expander.target,
                                 spaces=0)
    sig_flex = vflex @ expander @ flex
    shift = np.empty(expander.target.shape)
    shift[0] = (vol**2 / 12.)[..., None, None]
    shift[1] = 1
    shift = ift.makeField(expander.target, shift)
    if asperity is None:
        asp = ift.makeOp(shift.ptw("sqrt")) @ (freqxi*sig_flex)
    else:
        asp = ift.LognormalTransform(*asperity, prefix + 'asperity', 0)
        vasp = np.empty(dom.shape)
        vasp[0] = 1
        vasp[1] = 0
        vasp = ift.DiagonalOperator(ift.makeField(dom, vasp),
                                    domain=expander.target,
                                    spaces=0)
        sig_asp = vasp @ expander @ asp
        asp = freqxi*sig_flex*(ift.Adder(shift) @ sig_asp).ptw("sqrt")

    # FIXME shift, vasp, vflex have far more pixels than needed

    logsky = rve.IntWProcessInitialConditions(a0, b0, intop @ asp)

    rve.my_asserteq(logsky.target[1], ift.DomainTuple.make(domain)[0])
    rve.my_asserteq(logsky.target[0].size, nfreq)

    plotter.add_multiple2d("logsky", logsky)
    # FIXME Add all power spectra to plotter
    plotter.add_spectra("spectra", logsky, [[0.0002, 0.00035], [0.0004, 0.0001]])
    return logsky


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', type=int, default=1)
    parser.add_argument('--automatic-weighting', action='store_true')
    args = parser.parse_args()
    rve.set_nthreads(args.j)
    rve.set_wgridding(False)

    obs = rve.ms2observations('/data/CYG-D-6680-64CH-10S.ms', 'DATA', False, 0, 'stokesiavg')[0]

    # obs = rve.Observation.load('/data/g330field0.npz')

    print('Frequencies:')
    print(obs.freq)
    print('Shape visibilities:', obs.vis.shape)
    plt.scatter(np.arange(len(obs.freq)), obs.freq*1e-6)
    plt.ylabel('Frequency [MHz]')
    plt.xlabel('Channel')
    plt.savefig('debug_channels.png')
    plt.close()

    rve.set_epsilon(1/10/obs.max_snr())

    fov = np.array([3, 3])*rve.ARCMIN2RAD
    # Do not use powers of two here, otherwise super slow
    npix = np.array([250, 250])

    if False:
        R = rve.response.MfResponse(obs, rve.IRGSpace(obs.freq), ift.RGSpace(npix, fov/npix))
        j = R.adjoint(obs.vis*obs.weight)
        rve.plotter._plot_mf(rve.MinimizationState(j, []), ift.ScalingOperator(j.domain, 1), "out", None, 3)

    dom = ift.RGSpace(npix, fov/npix)
    plotter = rve.MfPlotter('png', 'plots')
    jump = 15
    logsky = mf_logsky(dom, obs.freq[jump//2::jump], 'sky', plotter)

    # Plot prior samples
    if False:
        for ii in range(3):
            state = rve.MinimizationState(ift.from_random(logsky.domain), [])
            plotter.plot(f"prior{ii}", state)

    sky = logsky.exp()

    if args.automatic_weighting:
        # FIXME Figure out how to do automatic weighting for mf
        npix = 2500
        effuv = np.linalg.norm(obs.effective_uvw(), axis=1)
        dom = ift.RGSpace(npix, 2*np.max(effuv)/npix)

        cfm = ift.CorrelatedFieldMaker.make(0, (2, 2), 'invcov', total_N=obs.nfreq)
        cfm.add_fluctuations(dom, (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5))
        logweighting = cfm.finalize(0)

        interpolation = rve.MfWeightingInterpolation(effuv, logweighting.target)
        weightop = ift.makeOp(obs.weight) @ (interpolation @ logweighting.exp())**(-2)
        lh_wgt = rve.MfImagingLikelihoodVariableCovariance(obs, sky, weightop)
        plotter.add_histogram('normalized residuals autowgts', lh_wgt.normalized_residual)

        # FIXME
        # plotter.add('bayesian weighting', logweighting.exp())
        plotter.add_multiple1d('power spectrum bayesian weighting', cfm.power_spectrum)
    lh = rve.MfImagingLikelihood(obs, sky)
    plotter.add_histogram('normalized residuals', lh.normalized_residual)

    minimizer = ift.NewtonCG(ift.GradientNormController(name='newton', iteration_limit=5))

    ham = ift.StandardHamiltonian(lh)
    state = rve.MinimizationState(0.1*ift.from_random(ham.domain), [])

    if args.automatic_weighting:
        for ii in range(5):
            state = rve.simple_minimize(ham, state.mean, 0, minimizer)
            plotter.plot(f"pre_sky_{ii}", state)

        lh = lh_wgt
        ham = ift.StandardHamiltonian(lh)
        pos = ift.MultiField.union([0.1*ift.from_random(ham.domain), state.mean])
        state = rve.MinimizationState(pos, [])

        for ii in range(5):
            state = rve.simple_minimize(ham, state.mean, 0, minimizer, constants=sky.domain.keys())
            plotter.plot(f"pre_wgt_{ii}", state)

    for ii in range(20):
        state = rve.simple_minimize(ham, state.mean, 0, minimizer)
        plotter.plot(f"iter{ii}", state)


if __name__ == '__main__':
    main()
