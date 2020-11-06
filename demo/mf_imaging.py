# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse

import matplotlib.pyplot as plt
import numpy as np

import nifty7 as ift
import resolve as rve


def mf_logsky(domain, freq, flexibility, asperity, prefix):
    # FIXME cumsum over first dimensions is performance-wise suboptimal
    assert np.all(np.diff(freq) > 0)  # need ordered frequencies

    assert len(set(np.diff(freq))) == 1
    f0 = freq[0]
    df = freq[1] - freq[0]
    freq_bounds = f0 - df/2. + df*np.arange(len(freq)+1)
    nfreq = len(freq)
    freqdom = rve.IRGSpace(freq_bounds)
    assert freqdom.size == nfreq

    # FIXME Take a0 and b0 into account
    a0 = ift.SimpleCorrelatedField(domain, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5),
                                   prefix=f'{prefix}a0')
    b0 = ift.SimpleCorrelatedField(domain, 0, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5),
                                   prefix=f'{prefix}b0')

    # IDEA Try to use individual power spectra
    # FIXME Support fixed variance for zero mode
    cfm = ift.CorrelatedFieldMaker.make(0., (1, 0.00001), f'{prefix}freqxi', total_N=2*(nfreq-1))
    # FIXME Support fixed fluctuations
    cfm.add_fluctuations(domain, (1, 0.00001), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5))
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
    vol = freqdom.dvol[:-1]  # FIXME Volume

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
    logsky = intop @ asp

    # pos = ift.from_random(logsky.domain)
    # from time import time
    # t0 = time()
    # logsky(pos)
    # print(f'{time()-t0}')

    # ift.extra.check_operator(logsky, ift.from_random(logsky.domain), ntries=10)

    rve.my_asserteq(logsky.target[1], ift.DomainTuple.make(domain)[0])
    rve.my_asserteq(logsky.target[0].size, nfreq)
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
    npix = np.array([256, 256])
    dom = ift.RGSpace(npix, fov/npix)

    # FIXME Add option to have fewer imaging bands than channels
    # FIXME Figure out why the values are so freaking big/small
    logsky = mf_logsky(dom, obs.freq, (1e-12, 1e-13), (1e14, 1e14), 'sky')
    sky = logsky.exp()

    for ii in range(3):
        rve.mf_plot(f'debug{ii}', logsky(ift.from_random(sky.domain)), 2)

    # FIXME Write mf likelihood
    # FIXME Figure out how to do automatic weighting for mf


def bench_cumsum_helper(npix, nfreq):
    from time import time
    a = np.zeros((2, nfreq, npix, npix))
    t0 = time()
    np.cumsum(a[0], axis=0)
    t1 = time()
    np.cumsum(a[1], axis=0)
    t2 = time()

    s0 = time()
    b = np.transpose(a, (0, 2, 3, 1))
    b = np.ascontiguousarray(b)
    s1 = time()
    np.cumsum(b[0], axis=-1)
    s2 = time()
    np.cumsum(b[1], axis=-1)
    s3 = time()
    b = np.transpose(a, (0, 3, 1, 2))
    b = np.ascontiguousarray(b)
    s4 = time()

    return t2-t0, s4-s0


def bench_cumsum():
    xs = [64, 70, 80, 93, 100, 128, 150, 189, 200, 256, 500, 512, 995, 999,
          1000, 1022, 1024, 2000, 2048]
    ys0 = []
    ys1 = []
    for ii, npix in enumerate(xs):
        print(ii, len(xs))
        nai, transp = bench_cumsum_helper(npix, 40)
        ys0.append(nai)
        ys1.append(transp)
    plt.plot(xs, ys0, label='Naive')
    plt.plot(xs, ys1, label='With transpose')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('bench_cumsum.png')
    print('Wrote bench_cumsum.png')


if __name__ == '__main__':
    bench_cumsum()
    main()
