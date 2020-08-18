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
    parser.add_argument('--automatic-weighting', action='store_true')
    parser.add_argument('ms')
    args = parser.parse_args()

    rve.set_nthreads(args.j)
    rve.set_wstacking(False)
    obs = rve.ms2observations(args.ms, 'DATA')[0].restrict_to_stokes_i()

    rve.set_epsilon(1/10/obs.max_snr())
    dst = 0.045*rve.AS2RAD
    dom = ift.RGSpace((4096, 2048), (dst, dst))
    sky = ift.SimpleCorrelatedField(dom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5)).exp()
    inserter = rve.PointInserter(sky.target, np.array([[0, 0], [0.7, -0.34]])*rve.AS2RAD)
    points = ift.InverseGammaOperator(inserter.domain, alpha=0.5, q=0.2/dom.scalar_dvol).ducktape('points')
    sky = sky + inserter @ points
    sky = rve.vla_beam(dom, np.mean(obs.freq)) @ sky

    ic = ift.AbsDeltaEnergyController(0.5, 3, 2000)
    minimizer = ift.NewtonCG(ift.GradientNormController(name='newton', iteration_limit=5))

    ham_imaging = ift.StandardHamiltonian(rve.ImagingLikelihood(obs, sky), ic)
    pos, _ = rve.simple_minimize(ham_imaging, pos, 0, minimizer)

    if args.automatic_weighting:
        invcovop = ift.InverseGammaOperator(obs.vis.domain, 1, 1/obs.weight).reciprocal().ducktape('invcov')
        ham = ift.StandardHamiltonian(rve.ImagingLikelihoodVariableCovariance(obs, sky, invcovop), ic)
        pos = pos.unite(ift.full(invcovop.domain, 0.))
        pos, _ = rve.simple_minimize(ham, pos, 0, minimizer)


if __name__ == '__main__':
    main()
