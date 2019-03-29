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

import numpy as np

import nifty5 as ift


def default_pspace(space):
    sp = ift.DomainTuple.make(space)
    if not len(sp) == 1:
        raise ValueError
    sp = sp[0].get_default_codomain()
    return ift.PowerSpace(sp)


def field2fits(field, file_name):
    from astropy.io import fits
    hdu = fits.PrimaryHDU()
    hdu.data = field.to_global_data()
    hdu.writeto(file_name, overwrite=True)


def calibrator_sky(domain, flux):
    middle_coords = tuple([i//2 for i in domain.shape])
    sky_c = np.zeros(domain.shape)
    sky_c[middle_coords] = flux
    return ift.Field.from_global_data(domain, sky_c).weight(-1)


def getfloat(cfg, key):
    return float(cfg.getfloat(key))


def getint(cfg, key):
    return int(cfg.getint(key))


def antennas(datahandlers):
    antennas = set()
    for dh in datahandlers:
        assert dh.ant1.min() >= 0
        antennas = antennas | set(dh.ant1) | set(dh.ant2)
    antennas = list(antennas)
    for dh in datahandlers:
        assert dh.ant1.max() <= len(antennas)
    antennas.sort()
    for ii, ant in enumerate(antennas):
        for dh in datahandlers:
            dh.ant1[dh.ant1 == ant] = ii
            dh.ant2[dh.ant2 == ant] = ii
    antennas = set()
    for dh in datahandlers:
        antennas = antennas | set(dh.ant1) | set(dh.ant2)
    return list(antennas)


def _calc_minmax(datahandlers):
    tmin, tmax = [], []
    for dh in datahandlers:
        tmin.append(dh.time.min())
        tmax.append(dh.time.max())
    return min(tmin), max(tmax)


def tmax(datahandlers):
    tmin, _ = _calc_minmax(datahandlers)
    for dh in datahandlers:
        dh.adjust_time(tmin)
    _, tmax = _calc_minmax(datahandlers)
    return tmax


def pickle(obj, fname):
    import pickle
    with open(fname, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(f):
    try:
        return np.load(f)
    except FileNotFoundError:
        raise FileNotFoundError('Input file not found.')
    except OSError:
        print(
            'Something went wrong during loading and unpickling of input file.'
        )
        exit()


def tuple_to_list(fld):
    dom = fld.domain[1]
    fld = fld.to_global_data()
    return [ift.from_global_data(dom, fld[ii]) for ii in range(fld.shape[0])]


def zero_to_nan(fld):
    dom = fld.domain
    fld = fld.to_global_data_rw()
    fld[fld == 0] = np.nan
    return ift.from_global_data(dom, fld)


def tuple_to_image(fld):
    dom = fld.domain
    assert len(dom.shape) == 2 and isinstance(
        dom[0], ift.UnstructuredDomain) and isinstance(dom[1], ift.RGSpace)
    vol = dom[1].shape[0]*dom[1].distances[0]
    newdom = ift.RGSpace(
        dom.shape[::-1], distances=[dom[1].distances[0], vol/dom.shape[0]/2])
    newdom = ift.RGSpace(dom.shape[::-1], distances=[dom[1].distances[0], 1.])
    return ift.from_global_data(newdom, fld.to_global_data().T)
