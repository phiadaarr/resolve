# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2019 Max-Planck-Society

from os.path import join

import numpy as np

import nifty5 as ift
import resolve as rve
from problem import Problem


def _print_section(s):
    print(80*'-')
    print(' {}'.format(s))
    print(80*'-')


if __name__ == '__main__':
    cfg_file = 'mock.cfg'
    np.seterr(all='raise', under='warn')
    np.random.seed(17)
    ift.fft.enable_fftw()

    cfg = rve.ConfigurationParser(cfg_file)

    plm = Problem(cfg)
    dhs = plm.dhs
    cal_ops = plm.cal_ops
    rs = plm.rs
    diffuse = plm.diffuse
    lh, lh_c = plm.lh, plm.lh_c

    rve.data_plot(join(cfg.out, 'dataplot'), dhs, 1, 3, 0, cal_ops)

    plot_overview = lambda position, name: rve.plot_overview(position, join(cfg.out, name), diffuse, None, diffuse, cal_ops, dhs, rs)

    try:
        plot_overview(plm.MOCK_POSITION, 'mock')
    except AttributeError:
        pass

    class Saver:
        def __init__(self, cfg):
            self._counter = 0
            self._cfg = cfg

        def save(self, position, name):
            rve.pickle(
                position,
                join(self._cfg.out, '{:02}_{}.pickle'.format(
                    self._counter, name)))
            plot_overview(position, '{:02}_{}'.format(self._counter, name))
            self._counter += 1

    def plot_samples(samples, name):
        p = ift.Plot()
        for samp in samples:
            p.add(diffuse.force(samp))
        p.output(name=join(cfg.out, name) + '.png', xsize=20, ysize=20)

    ###########################################################################
    # MINIMIZATION
    ###########################################################################
    saver = Saver(cfg)
    position = ift.full(lh.domain, 0.)

    plot_samples(
        [ift.from_random('normal', diffuse.domain) for _ in range(16)],
        'priorsamples')

    _print_section('Pre-calibration')
    for ii in range(cfg.min_pre_calib['sampling rounds']):
        minimizer = cfg.min_pre_calib['minimizer']
        c = cfg.min_pre_calib
        dct = {
            'position': position.extract(lh_c.domain),
            'ham': ift.StandardHamiltonian(lh_c, c['ic sampling']),
            'nsamp': c['n samples'],
        }
        e = rve.MetricGaussianKL(**dct)
        e, _ = minimizer(e)
        position = ift.MultiField.union([position, e.position])
        saver.save(position, 'precal')

    _print_section('Pre-imaging')
    for ii in range(4):
        _, lh_only_image = lh.simplify_for_constant_input(
            position.extract(lh_c.domain))
        minimizer = cfg.min_imaging['minimizer']
        pe = rve.key_handler.calib + rve.key_handler.zeromodes
        c = cfg.min_imaging
        dct = {
            'ham': ift.StandardHamiltonian(lh_only_image, c['ic sampling']),
            'point_estimates': pe,
            'nsamp': 3,
            'constants': rve.key_handler.calib
        }
        e = rve.MetricGaussianKL(position, **dct)
        plot_samples([samp + position for samp in e.samples],
                     'samples_{}'.format(ii))
        e, _ = minimizer(e)
        position = e.position
        saver.save(position, 'preimaging')

        # Adjust variances
        dct['nsamp'] = 3
        e = rve.MetricGaussianKL(position, **dct)
        plot_samples([samp + position for samp in e.samples],
                     'samples_adjvar_{}'.format(ii))
        position = diffuse.adjust_variances(position, minimizer, e.samples)
        saver.save(position, 'preimaging_adjvar')

    # Main imaging
    _print_section('Joint calibration and imaging')
    for ii in range(cfg.min_imaging['sampling rounds']):
        minimizer = cfg.min_imaging['minimizer']
        c = cfg.min_imaging
        dct = {
            'position': position,
            'ham': ift.StandardHamiltonian(lh, c['ic sampling']),
            'nsamp': c['n samples'],
        }
        e = rve.MetricGaussianKL(**dct)
        plot_samples([samp + position for samp in e.samples],
                     'samples_{}'.format(ii))
        e, _ = minimizer(e)
        position = e.position
        saver.save(position, 'imaging')

        # Adjust variances
        e = rve.MetricGaussianKL(**dct)
        plot_samples([samp + position for samp in e.samples],
                     'samples_adjvar_{}'.format(ii))
        position = diffuse.adjust_variances(position, minimizer, e.samples)
        saver.save(position, 'imaging_adjvar')
