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

def _ampl(key):
    return {
        'x': key + ' xi',
        't': key + ' tau',
        'p': key + ' phi',
        'zm': key + ' zeromode'
    }


def _cal(key):
    key = key + ' polarization '
    return {'p0': _ampl(key + '0'), 'p1': _ampl(key + '1')}


def _flatten_dictionary(d):
    res = {}
    for key, val in d.items():
        try:
            tmp = {}
            for k, v in val.items():
                tmp[key + ' ' + k] = v
            res = {**res, **_flatten_dictionary(tmp)}
        except AttributeError:
            res[key] = val
    return res


class KeyHandler:
    def __init__(self):
        self.all = {
            'diff': _ampl('diffuse'),
            'points': 'points',
            'cal': {
                'ph': _cal('phase'),
                'ampl': _cal('amplitude')
            },
        }
        flat = _flatten_dictionary(self.all)
        calib, sky, pspec, pol_xi, zeromodes = [], [], [], [], []
        calib_ampl = []
        for key, val in flat.items():
            spl = key.split()
            if spl[0] in ['diff', 'points']:
                sky.append(val)
            if spl[-1] in ['t', 'p', 'zm']:
                pspec.append(val)
            if spl[-1] == 'zm':
                zeromodes.append(val)
            if spl[0] == 'cal':
                calib.append(val)
                if spl[-1] == 'x':
                    pol_xi.append(val)
                if spl[1] == 'ampl':
                    calib_ampl.append(val)

        self.calib = calib
        self.sky = sky
        self.pspec = pspec
        self.zeromodes = zeromodes
        self.pol_xi = pol_xi
        self.calib_ampl = calib_ampl


key_handler = KeyHandler()
