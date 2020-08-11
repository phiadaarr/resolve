# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np
import pytest

import nifty7 as ift
import resolve as rve

pmp = pytest.mark.parametrize

nthreads = 2
ift.fft.set_nthreads(nthreads)
rve.set_nthreads(nthreads)
rve.set_epsilon(1e-4)
rve.set_wstacking(False)
OBSERVATION = rve.ms2observations('data/CYG-ALL-2052-2MHZ.ms', 'DATA')[0]
OBS = [OBSERVATION.restrict_to_stokes_i(), OBSERVATION.average_stokes_i()]
npix, fov = 256, 1*rve.DEG2RAD
dom = ift.RGSpace((npix, npix), (fov/npix, fov/npix))
sky0 = ift.SimpleCorrelatedField(dom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5)).exp()
sky = rve.vla_beam(dom, np.mean(OBSERVATION.freq)) @ sky0


@pmp('ms', ('CYG-ALL-2052-2MHZ', 'CYG-D-6680-64CH-10S', 'AM754_A030124_flagged'))
def test_save_and_load_hdf5(ms):
    spws = [None]
    if ms == 'AM754_A030124_flagged':
        spws = [0, 1]
        with pytest.raises(RuntimeError):
            rve.ms2observations(f'data/{ms}.ms', 'DATA', None)
    for spw in spws:
        obs = rve.ms2observations(f'data/{ms}.ms', 'DATA', spw)
        for ob in obs:
            print('Max SNR:', ob.max_snr())
            print('Fraction flagged:', ob.fraction_flagged())
            ob.save_to_hdf5('foo.hdf5')
            ob1 = rve.Observation.load_from_hdf5('foo.hdf5')
            assert ob == ob1
            ob1.compress()


def try_operator(op):
    pos = ift.from_random(op.domain)
    op(pos)
    op(ift.Linearization.make_var(pos))
    # ift.extra.check_operator(op, pos, ntries=10)


@pmp('obs', OBS)
def test_imaging_likelihood(obs):
    lh = rve.ImagingLikelihood(obs, sky)
    try_operator(lh)


@pmp('obs', OBS)
def test_varcov_likelihood(obs):
    invcovop = ift.InverseGammaOperator(obs.vis.domain, 1, 1/obs.weight).reciprocal().ducktape('invcov')
    lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, invcovop)
    try_operator(lh)


@pmp('obs', OBS)
@pmp('noisemodel', range(4))
def test_imaging_likelihoods(obs, noisemodel):
    efflen = obs.effective_uvwlen()
    npix = 2500
    dom = ift.RGSpace(npix, 2*max(efflen)/npix)
    baseline_distributor = ift.LinearInterpolator(dom, efflen.T)
    pol_freq_copy = ift.ContractionOperator(obs.vis.domain, (0, 2)).adjoint
    cf = ift.SimpleCorrelatedField(dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), 'invcov').exp()
    correction = pol_freq_copy @ baseline_distributor @ cf
    if noisemodel == 2:  # Multiplicative noise model TODO Try without **(-2)
        invcovop = ift.makeOp(obs.weight) @ correction**(-2)
    else:  # Additive noise model
        invcovop = (ift.Adder(1/obs.weight) @ correction**2).reciprocal()
    lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, invcovop)
    try_operator(lh)
