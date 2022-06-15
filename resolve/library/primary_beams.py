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
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np

from ..constants import ARCMIN2RAD, SPEEDOFLIGHT
from ..util import assert_sky_domain, my_assert
from .meerkat_beam import JimBeam


def _get_meshgrid(domain):
    nx, ny = domain.shape
    xf = domain.distances[0] * nx / 2
    yf = domain.distances[1] * ny / 2
    xs = np.linspace(-xf, xf, nx)
    ys = np.linspace(-yf, yf, ny)
    return np.meshgrid(xs, ys, indexing="ij")


def meerkat_beam(domain, freq, mode):
    """Return approximate version of MeerKAT beam

    Parameters
    ----------
    domain : RGSpace
        Domain on which the beam shall be defined.
    frequency : float
        Observing frequency in MHz.
    mode : str
        Either "L" (900 to 1650 MHz) or "UHF" (550 to 1050 MHz)
    """
    beam = JimBeam(f"MKAT-AA-{mode}-JIM-2020")
    xx, yy = _get_meshgrid(domain)
    res = beam.I(xx, yy, freq)
    assert res.shape == domain.shape
    return res


def mf_meerkat_beam(frequency_domain, spatial_domain, mode):
    """Return approximate version of MeerKAT beam

    Parameters
    ----------
    frequency_domain : IRGSpace

    spatial_domain : RGSpace

    mode : str
        Either "L" (900 to 1650 MHz) or "UHF" (550 to 1050 MHz)
    """
    freqs = frequency_domain.coordinates
    res = np.empty((len(freqs),) + spatial_domain.shape)
    for ii, freq in enumerate(freqs):
        res[ii] = meerkat_beam(spatial_domain, freq, mode)
    return res


def vla_beam_func(freq, x):
    freq = 1e-6 * float(freq)  # freq in MHz
    # Values taken from EVLA memo 195
    my_assert(freq < 14948, freq > 1040)
    coeffs = np.array(
        [
            [1040, -1.529, 8.69, -1.88],
            [1104, -1.486, 8.15, -1.68],
            [1168, -1.439, 7.53, -1.45],
            [1232, -1.450, 7.87, -1.63],
            [1296, -1.428, 7.62, -1.54],
            [1360, -1.449, 8.02, -1.74],
            [1424, -1.462, 8.23, -1.83],
            [1488, -1.455, 7.92, -1.63],
            [1552, -1.435, 7.54, -1.49],
            [1680, -1.443, 7.74, -1.57],
            [1744, -1.462, 8.02, -1.69],
            [1808, -1.488, 8.38, -1.83],
            [1872, -1.486, 8.26, -1.75],
            [1936, -1.469, 7.93, -1.62],
            [2000, -1.508, 8.31, -1.68],
            [2052, -1.429, 7.52, -1.47],
            [2180, -1.389, 7.06, -1.33],
            [2436, -1.377, 6.90, -1.27],
            [2564, -1.381, 6.92, -1.26],
            [2692, -1.402, 7.23, -1.40],
            [2820, -1.433, 7.62, -1.54],
            [2948, -1.433, 7.46, -1.42],
            [3052, -1.467, 8.05, -1.70],
            [3180, -1.497, 8.38, -1.80],
            [3308, -1.504, 8.37, -1.77],
            [3436, -1.521, 8.63, -1.88],
            [3564, -1.505, 8.37, -1.75],
            [3692, -1.521, 8.51, -1.79],
            [3820, -1.534, 8.57, -1.77],
            [3948, -1.516, 8.30, -1.66],
            [4052, -1.406, 7.41, -1.48],
            [4180, -1.385, 7.09, -1.36],
            [4308, -1.380, 7.08, -1.37],
            [4436, -1.362, 6.95, -1.35],
            [4564, -1.365, 6.92, -1.31],
            [4692, -1.339, 6.56, -1.17],
            [4820, -1.371, 7.06, -1.40],
            [4948, -1.358, 6.91, -1.34],
            [5052, -1.360, 6.91, -1.33],
            [5180, -1.353, 6.74, -1.25],
            [5308, -1.359, 6.82, -1.27],
            [5436, -1.380, 7.05, -1.37],
            [5564, -1.376, 6.99, -1.31],
            [5692, -1.405, 7.39, -1.47],
            [5820, -1.394, 7.29, -1.45],
            [5948, -1.428, 7.57, -1.57],
            [6052, -1.445, 7.68, -1.50],
            [6148, -1.422, 7.38, -1.38],
            [6308, -1.463, 7.94, -1.62],
            [6436, -1.478, 8.22, -1.74],
            [6564, -1.473, 8.00, -1.62],
            [6692, -1.455, 7.76, -1.53],
            [6820, -1.487, 8.22, -1.72],
            [6948, -1.472, 8.05, -1.67],
            [7052, -1.470, 8.01, -1.64],
            [7180, -1.503, 8.50, -1.84],
            [7308, -1.482, 8.19, -1.72],
            [7436, -1.498, 8.22, -1.66],
            [7564, -1.490, 8.18, -1.66],
            [7692, -1.481, 7.98, -1.56],
            [7820, -1.474, 7.94, -1.57],
            [7948, -1.448, 7.69, -1.51],
            [8052, -1.403, 7.21, -1.37],
            [8180, -1.398, 7.1, -1.32],
            [8308, -1.402, 7.16, -1.35],
            [8436, -1.4, 7.12, -1.32],
            [8564, -1.391, 6.95, -1.25],
            [8692, -1.409, 7.34, -1.49],
            [8820, -1.41, 7.36, -1.45],
            [8948, -1.41, 7.34, -1.43],
            [9052, -1.403, 7.2, -1.36],
            [9180, -1.396, 7.09, -1.31],
            [9308, -1.432, 7.68, -1.55],
            [9436, -1.414, 7.43, -1.47],
            [9564, -1.416, 7.45, -1.47],
            [9692, -1.406, 7.26, -1.39],
            [9820, -1.412, 7.36, -1.43],
            [9948, -1.409, 7.29, -1.39],
            [10052, -1.421, 7.46, -1.45],
            [10180, -1.409, 7.25, -1.36],
            [10308, -1.402, 7.13, -1.31],
            [10436, -1.399, 7.09, -1.29],
            [10564, -1.413, 7.37, -1.43],
            [10692, -1.412, 7.34, -1.41],
            [10820, -1.401, 7.12, -1.31],
            [10948, -1.401, 7.12, -1.31],
            [10152, -1.401, 7.12, -1.31],
            [11180, -1.394, 6.99, -1.24],
            [11308, -1.394, 7.01, -1.26],
            [11436, -1.391, 6.94, -1.22],
            [11564, -1.389, 6.92, -1.22],
            [11692, -1.386, 6.8, -1.15],
            [11820, -1.391, 6.88, -1.19],
            [11948, -1.399, 6.97, -1.22],
            [13308, -1.403, 7.37, -1.47],
            [13436, -1.392, 7.08, -1.31],
            [12052, -1.399, 7.17, -1.34],
            [12180, -1.392, 7.07, -1.31],
            [12308, -1.393, 7.19, -1.38],
            [12436, -1.393, 7.20, -1.4],
            [12564, -1.395, 7.19, -1.38],
            [12692, -1.397, 7.20, -1.37],
            [12820, -1.388, 7.06, -1.32],
            [12948, -1.397, 7.18, -1.36],
            [13052, -1.4, 7.27, -1.4],
            [13180, -1.406, 7.44, -1.5],
            [13308, -1.403, 7.37, -1.47],
            [13436, -1.392, 7.08, -1.31],
            [13564, -1.384, 6.94, -1.24],
            [13692, -1.382, 6.95, -1.25],
            [13820, -1.376, 6.88, -1.24],
            [13948, -1.384, 6.98, -1.28],
            [14052, -1.4, 7.36, -1.48],
            [14180, -1.397, 7.29, -1.45],
            [14308, -1.399, 7.32, -1.45],
            [14436, -1.396, 7.25, -1.42],
            [14564, -1.393, 7.3, -1.39],
            [14692, -1.384, 7.03, -1.31],
            [14820, -1.388, 7.06, -1.32],
            [14948, -1.393, 7.16, -1.37],
        ]
    ).T

    ind = np.argsort(coeffs[0])
    coeffs = coeffs[:, ind]

    freqs = coeffs[0]
    poly = coeffs[1:].T
    my_assert(freq <= freqs.max())
    my_assert(freq >= freqs.min())

    ind = np.searchsorted(freqs, freq)
    flower = freqs[ind - 1]
    clower = poly[ind - 1]
    fupper = freqs[ind]
    cupper = poly[ind]
    rweight = (freq - flower) / (fupper - flower)

    def _vla_eval_poly(coeffs, xs):
        my_assert(coeffs.shape == (3,))
        ys = np.ones_like(xs)
        ys += 1e-3 * coeffs[0] * xs ** 2
        ys += 1e-7 * coeffs[1] * xs ** 4
        return ys + 1e-10 * coeffs[2] * xs ** 6

    lower = _vla_eval_poly(clower, flower*x/1000/ARCMIN2RAD)
    upper = _vla_eval_poly(cupper, fupper*x/1000/ARCMIN2RAD)
    beam = rweight * upper + (1 - rweight) * lower
    beam[beam < 0] = 0
    return beam


def vla_beam(domain):
    from ..util import assert_sky_domain

    assert_sky_domain(domain)
    pdom, tdom, fdom, sdom = domain
    res = np.empty(ift.makeDomain((fdom, sdom)).shape)
    for ii, freq in enumerate(fdom.coordinates):
        res[ii] = _single_freq_vla_beam(sdom, freq)
    res = np.broadcast_to(res[None, None], domain.shape)
    return ift.makeOp(ift.makeField(domain, res))


def _single_freq_vla_beam(domain, freq):
    dom = ift.DomainTuple.make(domain)
    my_assert(len(dom) == 1)
    dom = dom[0]
    xx, yy = _get_meshgrid(dom)
    beam = vla_beam_func(freq, np.sqrt(xx ** 2 + yy ** 2))
    return beam


def alma_beam(domain, D, d):
    domain = ift.makeDomain(domain)
    assert_sky_domain(domain)
    _, _, fdom, sdom = domain
    npix = np.array(sdom.shape)[..., None, None]
    nx, ny = sdom.shape
    dst = np.array(sdom.distances)[..., None, None]
    length_arr = np.linalg.norm(dst * (np.mgrid[:nx, :ny] - 0.5*npix), axis=0)
    beam = np.empty(fdom.shape + sdom.shape)
    for ff, freq in enumerate(fdom.coordinates):
        beam[ff] = alma_beam_func(D, d, freq, length_arr)
    beam = ift.makeField((fdom, sdom), beam)
    return beam.ducktape_left(domain)


def alma_beam_func(D, d, freq, x, use_cache=False):
    assert isinstance(x, np.ndarray)
    assert x.ndim < 3
    assert np.max(np.abs(x)) < np.pi / np.sqrt(2)

    if not use_cache:
        return _compute_alma_beam(D, d, freq, x)

    iden = "_".join([str(ll) for ll in [D, d, freq]] + [str(ll) for ll in x.shape])
    fname = f".beamcache{iden}.npy"
    try:
        return np.load(fname)
    except FileNotFoundError:
        arr = _compute_alma_beam(D, d, freq, x)
        np.save(fname, arr)
        return arr


def _compute_alma_beam(D, d, freq, x):
    import scipy.special as sc

    a = freq / SPEEDOFLIGHT
    b = d / D
    x = np.pi * a * D * x
    mask = x == 0.0
    x[mask] = 1
    sol = 2 / (x * (1 - b ** 2)) * (sc.jn(1, x) - b * sc.jn(1, x * b))
    sol[mask] = 1
    return sol * sol
