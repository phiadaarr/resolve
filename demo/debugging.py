# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import resolve as rve
import nifty7 as ift

rve.set_nthreads(2)
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
    sky = rve.vla_beam(dom, 2e9) @ ift.SimpleCorrelatedField(dom, **args).exp()
    lh = rve.ImagingLikelihood(ob.restrict_to_stokes_i(), sky)
    lh = rve.ImagingLikelihood(ob.average_stokes_i(), sky)

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
