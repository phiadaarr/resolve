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


@pmp('mode', (True, False))
@pmp('noisemodel', range(4))
def test_imaging_likelihoods(mode, noisemodel):
    ob = OBSERVATION

    oo = ob.restrict_to_stokes_i() if mode else ob.average_stokes_i()
    if noisemodel == 0:
        lh = rve.ImagingLikelihood(oo, sky)
        try_operator(lh)
        return
    elif noisemodel == 1:
        invcovop = ift.InverseGammaOperator(oo.vis.domain, 1, 1/oo.weight).reciprocal().ducktape('invcov')
        lh = rve.ImagingLikelihoodVariableCovariance(oo, sky, invcovop)
        try_operator(lh)
        return
    efflen = oo.effective_uvwlen()
    npix = 2500
    dom = ift.RGSpace(npix, 2*max(efflen)/npix)
    baseline_distributor = ift.LinearInterpolator(dom, efflen.T)
    pol_freq_copy = ift.ContractionOperator(oo.vis.domain, (0, 2)).adjoint
    cf = ift.SimpleCorrelatedField(dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), 'invcov').exp()
    correction = pol_freq_copy @ baseline_distributor @ cf
    if noisemodel == 2:
        # Multiplicative noise model TODO Try without **(-2)
        invcovop = ift.makeOp(oo.weight) @ correction**(-2)
    else:
        # Additive noise model
        invcovop = (ift.Adder(1/oo.weight) @ correction**2).reciprocal()
    lh = rve.ImagingLikelihoodVariableCovariance(oo, sky, invcovop)
    try_operator(lh)
