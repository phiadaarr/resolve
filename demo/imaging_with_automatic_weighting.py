# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse

import numpy as np

import nifty7 as ift
import resolve as rve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', type=int, default=1)
    parser.add_argument('ms')
    args = parser.parse_args()

    rve.set_nthreads(args.j)
    rve.set_wgridding(False)
    obs = rve.ms2observations(args.ms, 'DATA', False, 0, 'stokesiavg')[0]

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
    npix = 2500
    effuv = np.linalg.norm(obs.effective_uv().T, axis=1)
    assert obs.nfreq == obs.npol == 1
    dom = ift.RGSpace(npix, 2*np.max(effuv)/npix)
    logweighting = ift.SimpleCorrelatedField(dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), 'invcov')
    interpolation = ift.LinearInterpolator(dom, effuv)
    weightop = ift.makeOp(obs.weight) @ (rve.AddEmptyDimension(interpolation.target) @ interpolation @ logweighting.exp())**(-2)
    lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, weightop)

    plotter = rve.Plotter('png', 'plots')
    plotter.add('logsky', logsky)
    plotter.add('power spectrum logsky', logsky.power_spectrum)
    plotter.add('bayesian weighting', logweighting.exp())
    plotter.add('power spectrum bayesian weighting', logweighting.power_spectrum)
    plotter.add('normalized residuals', lh.normalized_residual)

    ic = ift.AbsDeltaEnergyController(0.5, 3, 100, name='Sampling')
    minimizer = ift.NewtonCG(ift.GradientNormController(name='newton', iteration_limit=5))
    ham = ift.StandardHamiltonian(lh, ic)
    pos = 0.1*ift.from_random(ham.domain)
    state = rve.MinimizationState(pos, [])

    for ii in range(5):
        state = rve.simple_minimize(ham, state.mean, 5, minimizer)
        plotter.plot(f'both{ii}', state)


if __name__ == '__main__':
    main()
