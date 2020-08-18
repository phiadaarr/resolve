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
    dst = 0.045*rve.AS2RAD
    dom = ift.RGSpace((4096, 2048), (dst, dst))
    sky = ift.SimpleCorrelatedField(dom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5)).exp()
    inserter = rve.PointInserter(sky.target, np.array([[0, 0], [0.7, -0.34]])*rve.AS2RAD)
    points = ift.InverseGammaOperator(inserter.domain, alpha=0.5, q=0.2/dom.scalar_dvol).ducktape('points')
    sky = sky + inserter @ points
    sky = rve.vla_beam(dom, np.mean(obs.freq)) @ sky
    plotter = rve.Plotter('png', 'plots')
    plotter.add('logsky', sky.log())

    ic = ift.AbsDeltaEnergyController(0.5, 3, 2000)
    minimizer = ift.NewtonCG(ift.GradientNormController(name='newton', iteration_limit=5))

    ham_imaging = ift.StandardHamiltonian(rve.ImagingLikelihood(obs, sky), ic)
    if args.start is None:
        pos = 0.1*ift.from_random(ham_imaging.domain)
        plotter.plot('initial', pos)
        state = rve.simple_minimize(ham_imaging, pos, 0, minimizer)
        state = rve.simple_minimize(ham_imaging, state.mean, 0, minimizer)
        state = rve.simple_minimize(ham_imaging, state.mean, 0, minimizer)
        plotter.plot('imagingonly', state)
        state.save('currentstate')
        return
    alpha = 1
    invcovop = ift.InverseGammaOperator(obs.vis.domain, alpha, 1/obs.weight).reciprocal().ducktape('invcov')
    plotter.add_uvscatter('inverse covariance', invcovop, obs)
    ham = ift.StandardHamiltonian(rve.ImagingLikelihoodVariableCovariance(obs, sky, invcovop), ic)

    pos = rve.MinimizationState.load(args.start).mean.unite(ift.full(invcovop.domain, norm.ppf(invgamma.cdf(1, alpha))))
    plotter.plot('initial', pos)
    state = rve.simple_minimize(ham, pos, 0, minimizer, constants=sky.domain.keys())
    plotter.plot('learnnoise', state)
    state = rve.simple_minimize(ham, state.mean, 0, minimizer, constants=sky.domain.keys())
    plotter.plot('learnnoise1', state)
    state = rve.simple_minimize(ham, state.mean, 0, minimizer, constants=sky.domain.keys())
    plotter.plot('learnnoise2', state)
    state.save('learnnoise')


if __name__ == '__main__':
    main()
