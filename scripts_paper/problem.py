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
from time import time

import numpy as np

import nifty5 as ift
import resolve as rve
from powerspectrum import pspec
from resolve import getfloat, getint


class Problem:
    def __init__(self, cfg):
        snr = cfg.io.getfloat('snr')
        fieldid = cfg.io.getint('field id science target')
        dhs = {'t': rve.DataHandler(cfg, fieldid)}
        fieldid = cfg.io.getint('field id phase calibrator')
        dhs['c'] = rve.DataHandler(cfg, fieldid)

        tmax = rve.tmax(dhs.values())
        antennas = rve.antennas(dhs.values())
        print('Length of observation: {0:.1f} min'.format(tmax/60))

        c = cfg.diffuse
        keys = rve.key_handler.all['diff']
        dct = {
            'target': rve.default_pspace(cfg.sky_space),
            'n_pix': getint(c, 'pix dof space'),
            'a': getfloat(c, 'a'),
            'k0': getfloat(c, 'k0'),
            'sm': getfloat(c, 'slope mean'),
            'sv': getfloat(c, 'slope variance'),
            'im': getfloat(c, 'y-intercept mean'),
            'iv': getfloat(c, 'y-intercept variance'),
            'keys': [keys['t'], keys['p']]
        }
        a = rve.SLAmplitude(
            slamplcfg=dct,
            alpha=getfloat(cfg.diffuse, 'log-zeromode variance alpha'),
            q=getfloat(cfg.diffuse, 'log-zeromode variance q'),
            zeromode_key=keys['zm'])
        dct = {'target': cfg.sky_space, 'amplitude': a}
        diffuse = rve.Diffuse(**dct)

        cal_ops = {}
        if cfg.ampl_calib:
            c = cfg.ampl_calib
            amplcfg = {
                'n_pix': getint(c, 'pix dof space'),
                'sm': getfloat(c, 'slope mean'),
                'sv': getfloat(c, 'slope variance'),
                'im': getfloat(c, 'y-intercept mean'),
                'iv': getfloat(c, 'y-intercept variance'),
                'alpha': getfloat(c, 'log-zeromode variance alpha'),
                'q': getfloat(c, 'log-zeromode variance q'),
                'linear_key': rve.key_handler.all['cal']['ampl']['p0']['p'],
                'zeromode_key': rve.key_handler.all['cal']['ampl']['p0']['zm']
            }
            dct = {
                't_pix': cfg.time_pix,
                't_max': tmax,
                'antennas': antennas,
                'xi_key': rve.key_handler.all['cal']['ampl']['p0']['x'],
                'amplitude': amplcfg,
                'zero_padding_factor': 2,
                'clip': [-10, 10]
            }
            cal_ops['ampl0'] = rve.Calibration(**dct)

            dct['amplitude'] = cal_ops['ampl0'].amplitude
            dct['xi_key'] = rve.key_handler.all['cal']['ampl']['p1']['x']
            cal_ops['ampl1'] = rve.Calibration(**dct)
        if cfg.phase_calib:
            c = cfg.phase_calib
            amplcfg = {
                'n_pix': getint(c, 'pix dof space'),
                'sm': getfloat(c, 'slope mean'),
                'sv': getfloat(c, 'slope variance'),
                'im': getfloat(c, 'y-intercept mean'),
                'iv': getfloat(c, 'y-intercept variance'),
                'alpha': getfloat(c, 'log-zeromode variance alpha'),
                'q': getfloat(c, 'log-zeromode variance q'),
                'linear_key': rve.key_handler.all['cal']['ph']['p0']['p'],
                'zeromode_key': rve.key_handler.all['cal']['ph']['p0']['zm'],
            }
            dct = {
                't_pix': cfg.time_pix,
                't_max': tmax,
                'antennas': antennas,
                'xi_key': rve.key_handler.all['cal']['ph']['p0']['x'],
                'amplitude': amplcfg,
                'zero_padding_factor': 2,
                'clip': [-15, 15]
            }
            cal_ops['ph0'] = rve.Calibration(**dct)

            dct['amplitude'] = cal_ops['ph0'].amplitude
            dct['xi_key'] = rve.key_handler.all['cal']['ph']['p1']['x']
            cal_ops['ph1'] = rve.Calibration(**dct)

        flux = float(cfg.likelihood.getfloat('calibrator flux'))
        calib_sky = rve.calibrator_sky(diffuse.target, flux)

        dct = {
            'wplanes': getint(cfg.response, 'w planes'),
            'domain': cfg.sky_space
        }
        rs = {
            't': rve.Response(dhs['t'], **dct),
            'c': rve.Response(dhs['c'], **dct)
        }

        if snr is not None:
            # np.random.seed(42)
            harmonic_space = diffuse._xi.target[0]
            power_space = ift.PowerSpace(harmonic_space)
            PD = ift.PowerDistributor(harmonic_space, power_space)
            prior_correlation_structure = PD(ift.PS_field(power_space, pspec))
            S = ift.DiagonalOperator(prior_correlation_structure)
            HT = ift.HarmonicTransformOperator(
                harmonic_space, target=diffuse.target[0])

            mock_sky = HT(S.draw_sample()).exp()
            MOCK_POSITION = diffuse.pre_image(mock_sky)
            mock_sky2 = diffuse(MOCK_POSITION)
            np.testing.assert_almost_equal(mock_sky.to_global_data(),
                                           mock_sky2.to_global_data())
            st0 = np.random.get_state()
            np.random.seed(5)
            caldom = ift.MultiDomain.union(
                [op.domain for op in cal_ops.values()])
            MOCK_POSITION = ift.MultiField.union(
                [ift.from_random('normal', caldom), MOCK_POSITION])
            rve.plot_overview(MOCK_POSITION, 'mock', diffuse, None, diffuse,
                              cal_ops, dhs, rs)
            rve.pickle(MOCK_POSITION, join(cfg.out, 'mock.pickle'))

            dmodel = rve.make_signal_response(dhs['t'], rs['t'], diffuse,
                                              cal_ops)
            d_mock = dmodel.force(MOCK_POSITION)
            dhs['t'].var = ift.full(dhs['t'].var.domain,
                                    d_mock.to_global_data().var()/snr)
            N = ift.makeOp(dhs['t'].var)
            dhs['t'].vis = d_mock + N.draw_sample() + 1j*N.draw_sample()

            dmodel = rve.make_signal_response(dhs['c'], rs['c'], calib_sky,
                                              cal_ops)
            d_mock = dmodel.force(MOCK_POSITION)
            dhs['c'].var = ift.full(dhs['c'].var.domain,
                                    d_mock.to_global_data().var()/snr)
            N = ift.makeOp(dhs['c'].var)
            dhs['c'].vis = d_mock + N.draw_sample() + 1j*N.draw_sample()
            np.random.set_state(st0)
            self.MOCK_POSITION = MOCK_POSITION

        lh = rve.make_likelihood(dhs['t'], rs['t'], diffuse, cal_ops)
        lh_c = rve.make_likelihood(dhs['c'], rs['c'], calib_sky, cal_ops)
        lh = lh + lh_c

        sc = ift.StatCalculator()
        for _ in range(10):
            fld = ift.from_random('normal', lh.domain)
            t0 = time()
            lh(fld)
            sc.add(time() - t0)
        print('One likelihood evaluation takes: ({:.1e} +/- {:.1e}) s'.format(
            sc.mean, np.sqrt(sc.var)))

        self.dhs = dhs
        self.cal_ops = cal_ops
        self.rs = rs
        self.diffuse = diffuse
        self.lh, self.lh_c = lh, lh_c
