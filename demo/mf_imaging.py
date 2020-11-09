# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse

import matplotlib.pyplot as plt
import numpy as np

import nifty7 as ift
import resolve as rve


def mf_logsky(domain, freq, prefix, plotter):
    # FIXME cumsum over first dimensions may be performance-wise suboptimal
    assert np.all(np.diff(freq) > 0)  # need ordered frequencies

    # freq = freq[::5]  # Fewer imaging bands, FIXME write interface
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

    # FIXME Move to tests
    # FIXME Make test working
    # ift.extra.check_linear_operator(intop)
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
    pos = ift.from_random(logsky.domain)
    out = logsky(pos)
    # FIXME Move to tests
    # FIXME Write also test which tests first bin from explicit formula
    np.testing.assert_equal(out.val[0], a0.force(pos).val)

    # ift.extra.check_operator(logsky, ift.from_random(logsky.domain), ntries=10)

    rve.my_asserteq(logsky.target[1], ift.DomainTuple.make(domain)[0])
    rve.my_asserteq(logsky.target[0].size, nfreq)

    plotter.add_mf("logsky", logsky)
    plotter.add_spectra("spectra", logsky, [[0.0002, 0.00035], [0.0004, 0.0001]])
    return logsky


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', type=int, default=1)
    args = parser.parse_args()
    rve.set_nthreads(args.j)
    rve.set_wgridding(False)

    obs = rve.ms2observations('/data/CYG-D-6680-64CH-10S.ms', 'DATA', False, 0, 'stokesiavg')[0]

    # obs = rve.Observation.load_from_npz('/data/g330field0.npz')

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
    dom = ift.RGSpace(npix, fov/npix)

    plotter = rve.MfPlotter('png', 'plots')

    logsky = mf_logsky(dom, obs.freq, 'sky', plotter)

    # Plot prior samples
    if False:
        for ii in range(3):
            state = rve.MinimizationState(ift.from_random(logsky.domain), [])
            plotter.plot(f"prior{ii}", state)

    sky = logsky.exp()

    # FIXME Figure out how to do automatic weighting for mf
    lh = rve.MfImagingLikelihood(obs, sky)
    plotter.add_histogram('normalized residuals', lh.normalized_residual)

    ham = ift.StandardHamiltonian(lh)
    pos = 0.1*ift.from_random(ham.domain)
    state = rve.MinimizationState(pos, [])
    minimizer = ift.NewtonCG(ift.GradientNormController(name='newton', iteration_limit=10))

    for ii in range(20):
        state = rve.simple_minimize(ham, state.mean, 0, minimizer)
        plotter.plot(f"iter{ii}", state)


if __name__ == '__main__':
    main()
