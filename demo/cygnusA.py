# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import sys

import numpy as np

import nifty7 as ift
import resolve as rve


def main():
    _, ms = sys.argv
    nthreads = 8
    ift.fft.set_nthreads(nthreads)
    rve.set_nthreads(nthreads)
    rve.set_epsilon(1e-4)
    rve.set_wstacking(False)
    obs = rve.ms2observations(ms, 'DATA')[0].restrict_to_stokes_i()
    dst = 0.045*rve.AS2RAD
    dom = ift.RGSpace((4096, 2048), (dst, dst))
    sky = ift.SimpleCorrelatedField(dom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5)).exp()
    inserter = rve.PointInserter(sky.target, np.array([[0, 0], [0.7, -0.34]])*rve.AS2RAD)
    points = ift.InverseGammaOperator(inserter.domain, alpha=0.5, q=0.2/dom.scalar_dvol).ducktape('points')
    sky = sky + inserter @ points
    sky = rve.vla_beam(dom, np.mean(obs.freq)) @ sky
    imaging_lh = rve.ImagingLikelihood(obs, sky)

    ham = ift.StandardHamiltonian(imaging_lh, ift.AbsDeltaEnergyController(0.5, 3, 2000))
    ham = ift.EnergyAdapter(0.1*ift.from_random(imaging_lh.domain), ham, want_metric=True)
    minimizer = ift.NewtonCG(ift.GradientNormController(name='newton', iteration_limit=5))
    ham, _ = minimizer(ham)
    # invcovop = ift.InverseGammaOperator(obs.vis.domain, 1, 1/obs.weight).reciprocal().ducktape('invcov')
    # full_lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, invcovop)


if __name__ == '__main__':
    main()
