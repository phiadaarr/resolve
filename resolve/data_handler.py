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

from os.path import isdir, join

import numpy as np
from casacore.tables import table

import nifty5 as ift

from .constants import SPEEDOFLIGHT


def _extend(a, n, axis=2):
    if axis == 2:
        return np.repeat(a[:, :, None], n, axis=2)
    if axis == 1:
        return np.repeat(a[:, None], n, axis=1)
    raise NotImplementedError


def _apply_mask(mask, dct):
    return {key: arr[mask] for key, arr in dct.items()}


class DataHandler:
    def __init__(self, cfg, fieldid):
        c = cfg.io
        dcol = str(c.get('data column'))
        spw = c.getint('spectral window')
        ch = c.getint('channel')
        ndata = c.getint('number random rows')
        ms = cfg.ms
        if not isdir(ms):
            raise RuntimeError('Input not found: {}'.format(ms))

        # Load data
        t = table(cfg.ms, readonly=True)
        d = {
            'uvw': t.getcol("UVW"),  # [:, uvw]
            'vis': np.array(t.getcol(dcol),
                            dtype=np.complex128),  # [:, ch, corr]
            'var': t.getcol("SIGMA")**2,  # [:, corr]
            'flag': np.array(t.getcol('FLAG'), np.bool),  # [:, ch, corr]
            'data_desc_id': t.getcol('DATA_DESC_ID'),  # [:]
            'field': t.getcol('FIELD_ID'),  # [:]
            'ant1': t.getcol('ANTENNA1'),  # [:]
            'ant2': t.getcol('ANTENNA2'),  # [:]
            'time': t.getcol('TIME')  # [:]
        }
        t.close()

        t = table(join(ms, 'SPECTRAL_WINDOW'), readonly=True)
        freqs = t.getcol('CHAN_FREQ')
        t.close()

        t = table(join(ms, 'POLARIZATION'), readonly=True)
        corr_types = list(t.getcol('CORR_TYPE')[0])
        t.close()

        # Select polarization
        # RR (5), RL (6), LR (7), and LL (8) for circular polarization
        # XX (9), XY (10), YX (11), and YY (12) for linear polarization
        for ps in [[5, 8], [9, 12]]:
            if set(ps) <= set(corr_types):
                pols = [corr_types.index(ps[ii]) for ii in range(2)]
                break
        d['vis'] = d['vis'][:, :, pols]
        d['flag'] = d['flag'][:, :, pols]
        d['var'] = d['var'][:, pols]

        # Select channel
        if ch > 0:
            d['vis'] = d['vis'][:, ch:ch + 1]
            d['flag'] = d['flag'][:, ch:ch + 1]
            freqs = freqs[:, ch:ch + 1]
            print(d['vis'].shape)
            if d['vis'].shape[1] == 0:
                raise RuntimeError('Channel not valid.')

        # Select spectral window
        d = _apply_mask(d['data_desc_id'] == spw, d)
        freqs = freqs[spw:spw + 1]
        del (d['data_desc_id'])

        # Apply flags
        # Data are flagged bad if the FLAG array element is True.
        # https://casa.nrao.edu/Memos/229.html
        # A data point is only taken if all correlations are not flagged.
        d['flag'] = np.any(d['flag'], axis=2)

        # Delete row if all channels are flagged.
        d = _apply_mask(~np.all(d['flag'][:, :], axis=1), d)

        # Select field
        d = _apply_mask(d['field'] == fieldid, d)
        if d['vis'].shape[0] == 0:
            raise RuntimeError('No data points left after selecting field.')
        del (d['field'])

        # Broadcast channel dimension and apply frequency to data
        nch = freqs.shape[1]
        lam = SPEEDOFLIGHT/freqs[0]
        uvw = _extend(d['uvw'], nch)/lam
        uvw = np.transpose(uvw, (0, 2, 1))  # [:, ch, uvw]
        del (d['uvw'])
        d['uv'], d['w'] = uvw[:, :, 0:2], uvw[:, :, 2]
        d['var'] = _extend(d['var'], nch, axis=1)

        # Remove too long baselines
        dst = cfg.sky_space.distances[0]
        mask = np.any(np.abs(d['uv']) > .5/dst, axis=2)
        d['flag'] = d['flag'] | mask

        # Select subset of data points
        len_vis = d['vis'].shape[0]
        if ndata == -1 or (ndata > 0 and ndata > len_vis):
            print('Take all data points.')
        else:
            st0 = np.random.get_state()
            d = _apply_mask(np.random.choice(len_vis, ndata), d)
            np.random.set_state(st0)

        shp = d['vis'].shape
        row_space = ift.UnstructuredDomain(shp[0])
        channel_space = ift.UnstructuredDomain(shp[1])
        polarization_space = ift.UnstructuredDomain(shp[2])
        d_space = ift.DomainTuple.make((row_space, channel_space,
                                        polarization_space))

        self.uv = d['uv']
        self.w = d['w']
        self.vis = ift.from_global_data(d_space, d['vis'])

        # Factor 2: Suppose noise is on real and imaginary part each. Then the
        # complex number has twice the variance.
        self.var = 2*ift.from_global_data(d_space, d['var'])
        self.ant1 = d['ant1']
        self.ant2 = d['ant2']
        self.time = d['time']
        self.flag = d['flag']
        self.freqs = freqs

    def adjust_time(self, tmin):
        if self.time.min() < tmin:
            raise ValueError
        self.time = self.time - tmin
