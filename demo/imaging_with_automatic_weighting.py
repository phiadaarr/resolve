# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse

import numpy as np
from scipy.stats import invgamma, norm

import nifty7 as ift
import resolve as rve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', type=int, default=1)
    parser.add_argument('--automatic-weighting', action='store_true')
    parser.add_argument('--start')
    parser.add_argument('ms')
    args = parser.parse_args()

    rve.set_nthreads(args.j)
    rve.set_wstacking(False)
    # obs = rve.ms2observations(args.ms, 'DATA')[0].average_stokes_i()
    obs = rve.ms2observations(args.ms, 'DATA')[0].restrict_to_stokes_i()

    rve.set_epsilon(1/10/obs.max_snr())
    fov = np.array([3, 1.5])*rve.ARCMIN2RAD
    npix = np.array([4096, 2048])/4
    dom = ift.RGSpace(npix, fov/npix)
    logsky = ift.SimpleCorrelatedField(dom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5))
    sky = logsky.exp()
    inserter = rve.PointInserter(sky.target, np.array([[0, 0], [0.7, -0.34]])*rve.AS2RAD)
    points = ift.InverseGammaOperator(inserter.domain, alpha=0.5, q=0.2/dom.scalar_dvol).ducktape('points')
    sky = sky + inserter @ points
    sky = rve.vla_beam(dom, np.mean(obs.freq)) @ sky
    alpha = 1
    invcovop = ift.InverseGammaOperator(obs.vis.domain, alpha, 1/obs.weight).reciprocal().ducktape('invcov')
    plotter = rve.Plotter('png', 'plots')
    plotter.add('logsky', logsky)
    plotter.add('power spectrum logsky', logsky.power_spectrum)
    plotter.add_uvscatter('inverse covariance', invcovop, obs)

    ic = ift.AbsDeltaEnergyController(0.5, 3, 100, name='Sampling')
    minimizer = ift.NewtonCG(ift.GradientNormController(name='newton', iteration_limit=5))

    ham = ift.StandardHamiltonian(rve.ImagingLikelihoodVariableCovariance(obs, sky, invcovop), ic)

    pos = ift.MultiField.union([0.1*ift.from_random(ham.domain), ift.full(invcovop.domain, norm.ppf(invgamma.cdf(1, alpha)))])
    state = rve.MinimizationState(pos, [])

    keys0 = sky.domain.keys()
    keys1 = invcovop.domain.keys()

    plotter.plot('initial', state)
    state = rve.simple_minimize(ham, state.mean, 0, minimizer, constants=keys1)
    plotter.plot('initial1', state)
    state = rve.simple_minimize(ham, state.mean, 0, minimizer, constants=keys1)
    plotter.plot('initialimaging', state)
    for ii in range(5):
        state = rve.simple_minimize(ham, state.mean, 0, minimizer, constants=keys0)
        plotter.plot(f'noise{ii}', state)
        state = rve.simple_minimize(ham, state.mean, 0, minimizer, constants=keys1)
        plotter.plot(f'sky{ii}', state)
    for ii in range(5):
        state = rve.simple_minimize(ham, state.mean, 0, minimizer)
        plotter.plot(f'both{ii}', state)


if __name__ == '__main__':
    main()
