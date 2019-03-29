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

import configparser
import datetime
from os import makedirs
from os.path import abspath, exists, expanduser, expandvars, join

import nifty5 as ift

from .constants import ARCMIN2RAD
from .sugar import getfloat, getint
from .version import gitversion


def _enabled(cfg):
    if cfg.getboolean('enable'):
        return cfg
    return False


def _parse_minimizer_settings(cfg_section):
    ic = ift.GradInfNormController(
        name='NewtonCG',
        tol=getfloat(cfg_section, 'tolerance'),
        iteration_limit=getint(cfg_section, 'maxsteps'))
    res = {'minimizer': ift.NewtonCG(ic)}
    res['sampling rounds'] = getint(cfg_section, 'sampling rounds')
    res['ic sampling'] = ift.GradientNormController(
        iteration_limit=getint(cfg_section, 'cg sampling'))
    res['n samples'] = getint(cfg_section, 'n samples')
    return res


class ConfigurationParser:
    '''
    Parameters
    ----------
    config_file : string
        Path to configuration file which is in the standard format for
        python's `configparser`.

    Attributes
    ----------
    out : string
        Path for storing results.
    sky_space : nifty.RGSpace
        Regular grid space on which the sky reconstruction is defined. The
        pixel size of this space is measured in radian.
    diffuse : configparser.SectionProxy
        Configuration of diffuse sky component prior.
    points : configparser.SectionProxy
        Configuration of point-like sky component prior.
    phase_calib : configparser.SectionProxy
        Configuration of phase calibration prior.
    ampl_calib : configparser.SectionProxy
        Configuration of amplitude calibration prior.
    response : configparser.SectionProxy
        Configuration of response.
    '''

    def __init__(self, config_file):
        # Load config file
        cfg = configparser.ConfigParser()
        if not exists(config_file):
            raise RuntimeError('Config file not found')
        cfg.read(config_file)
        self._cfg = cfg

        # Private sections
        cfg_dom = cfg['Domains']
        # Public sections
        self.io = cfg['IO']
        self.diffuse = _enabled(cfg['Diffuse sky component'])
        self.points = _enabled(cfg['Point-like sky component'])
        self.phase_calib = _enabled(cfg['Phase calibration'])
        self.ampl_calib = _enabled(cfg['Amplitude calibration'])
        self.response = cfg['Response']
        self.likelihood = cfg['Likelihood']

        # Set up output path
        t = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        default = '{}_{}'.format(t, gitversion())
        self.out = abspath(
            expandvars(expanduser(self.io.get('output dir', default))))
        self.ms = abspath(
            expandvars(expanduser(self.io.get('input', default))))
        if self.ms[-1] == '/':
            # FIXME Is this really needed?
            self.ms = self.ms[:-1]

        pix = cfg_dom.getint('sky pix')
        fov = float(cfg_dom.getfloat('field of view'))
        dst = fov*ARCMIN2RAD/pix
        self.sky_space = ift.RGSpace(2*(pix,), distances=2*(dst,))
        self.time_pix = getint(cfg_dom, 'time space pix')

        # FIXME Check compatibility
        # fastRESOLVE and noise estimation/calibration do not work
        # simultaneously
        # diffuse or points must be

        self.min_pre_calib = _parse_minimizer_settings(
            cfg['Minimization pre-calibration'])
        self.min_imaging = _parse_minimizer_settings(
            cfg['Minimization imaging'])

        keys = []
        if self.ampl_calib:
            keys.append('ampl')
        if self.phase_calib:
            keys.append('ph')
        self.keys = keys

        if not exists(self.out):
            makedirs(self.out)
        fname = join(self.out, 'versions.txt')
        # Save version information and config file
        ss = 'resolve: {}\n'.format(gitversion())
        ss += 'NIFTy version: {}\n'.format(ift.version.gitversion())
        with open(fname, 'w') as f:
            f.write(ss)
        fname = join(self.out, 'config.txt')
        with open(fname, 'w') as f:
            self._cfg.write(f)
