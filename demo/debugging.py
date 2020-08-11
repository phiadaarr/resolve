# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift
import resolve as rve

nthreads = 2
ift.fft.set_nthreads(nthreads)
rve.set_nthreads(nthreads)
rve.set_epsilon(1e-4)
rve.set_wstacking(False)


def save_and_load_hdf5(obs):
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


def main():
    ob = rve.ms2observations('./CYG-ALL-2052-2MHZ.ms', 'DATA')[0]
    args = {'offset_mean': 0,
            'offset_std': (1e-3, 1e-6),
            'fluctuations': (2., 1.),
            'loglogavgslope': (-4., 1),
            'flexibility': (5, 2.),
            'asperity': (0.5, 0.5)}
    npix, fov = 256, 1*rve.DEG2RAD
    dom = ift.RGSpace((npix, npix), (fov/npix, fov/npix))
    sky = rve.vla_beam(dom, np.mean(ob.freq)) @ ift.SimpleCorrelatedField(dom, **args).exp()

    for oo in (ob.restrict_to_stokes_i(), ob.average_stokes_i()):
        lh = rve.ImagingLikelihood(oo, sky)
        try_operator(lh)

        invcov0 = ift.InverseGammaOperator(oo.vis.domain, 1, 1/oo.weight).reciprocal().ducktape('invcov')
        efflen = oo.effective_uvwlen()
        npix = 2500
        dom = ift.RGSpace(npix, 2*max(efflen)/npix)
        cf = ift.SimpleCorrelatedField(dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), 'invcov').exp()
        baseline_distributor = ift.LinearInterpolator(dom, efflen.T)
        pol_freq_copy = ift.ContractionOperator(oo.vis.domain, (0, 2)).adjoint
        correction = pol_freq_copy @ baseline_distributor @ cf
        # Multiplicative noise model TODO Try without **(-2)
        invcov1 = ift.makeOp(oo.weight) @ correction**(-2)
        # Additive noise model
        invcov2 = (ift.Adder(1/oo.weight) @ correction**2).reciprocal()
        for invcovop in [invcov0, invcov1, invcov2]:
            lh = rve.ImagingLikelihoodVariableCovariance(oo, sky, invcovop)
            try_operator(lh)

    obs = rve.ms2observations('./CYG-ALL-2052-2MHZ.ms', 'DATA')
    save_and_load_hdf5(obs)
    obs = rve.ms2observations('./CYG-D-6680-64CH-10S.ms', 'DATA')
    save_and_load_hdf5(obs)
    obs = rve.ms2observations('./AM754_A030124_flagged.ms', 'DATA', 0)
    save_and_load_hdf5(obs)
    obs = rve.ms2observations('./AM754_A030124_flagged.ms', 'DATA', 1)
    save_and_load_hdf5(obs)


if __name__ == '__main__':
    main()
