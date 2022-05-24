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
# Copyright(C) 2019-2022 Max-Planck-Society
# Author: Philipp Arras

import time
from os.path import splitext

import nifty8 as ift
import numpy as np

from ..constants import DEG2RAD
from ..util import assert_sky_domain
from ..data.observation import Observation


def field2fits(field, file_name, observations=[], header_override={}):
    import astropy.io.fits as pyfits
    from astropy.time import Time
    
    if isinstance(observations, Observation):
        observations = [observations]
    if len(observations) == 0:
        direction = None
    else:
        if len(observations) > 1:
            warn("field2fits: Include only info of first observation into fits file.")
        direction = observations[0].direction

    dom = field.domain
    assert_sky_domain(dom)
    pdom, tdom, fdom, sdom = dom
    if tdom.size > 1:
        raise NotImplementedError
    h = pyfits.Header()
    h["ORIGIN"] = "resolve"
    h["BUNIT"] = "Jy/sr"
    # h["BTYPE"] = "INTENSITY"
    # h["BSCALE"] = 1.
    # h["BZERO"] = 0.
    # h["BMAJ"] = 0.
    # h["BMIN"] = 0.
    # h["BPA"] = 0.
    # h["DATE-OBS"] = '2019-05-08T20:32:19.1'  # TEMPORARY
    # #h["TELESCOPE"] = "MEERKAT" # TEMPORARY
    # h["OBSERVER"] = "xxx" # TEMPORARY
    h["OBSRA"] = direction.phase_center[0] * 180 / np.pi if direction is not None else 0.0
    h["OBSDEC"] = direction.phase_center[1] * 180 / np.pi if direction is not None else 0.0
    # h["SPECSYS"] = "TOPOCENT"
    h["CTYPE1"] = "RA---SIN"
    h["CRVAL1"] = direction.phase_center[0] * 180 / np.pi if direction is not None else 0.0
    h["CDELT1"] = -sdom.distances[0] * 180 / np.pi
    h["CRPIX1"] = sdom.shape[0] / 2
    h["CUNIT1"] = "deg"
    h["CTYPE2"] = "DEC--SIN"
    h["CRVAL2"] = direction.phase_center[1] * 180 / np.pi if direction is not None else 0.0
    h["CDELT2"] = sdom.distances[1] * 180 / np.pi
    h["CRPIX2"] = sdom.shape[1] / 2
    h["CUNIT2"] = "deg"

    df = fdom.distances
    if fdom.regular:
        f0 = fdom.coordinates[0]
        df = fdom.distances[0] if fdom.size != 1 else 1.
    else:
        warn("Cannot include frequency information into FITS because frequency domain is not uniformly spaced.")
        f0 = 1.
        df = 1.
    h["CTYPE3"] = "FREQ"
    h["CRVAL3"] = f0
    h["CDELT3"] = df
    h["CRPIX3"] = "0"
    h["CUNIT3"] = "Hz"
    h["CTYPE4"] = "STOKES"
    h["CRVAL4"] = 1.
    h["CDELT4"] = 1.
    h["CRPIX4"] = 1.
    h["CUNIT4"] = ""
    h["NAXIS"] = 4
    h["NAXIS1"] = sdom.shape[0]
    h["NAXIS2"] = sdom.shape[1]
    h["NAXIS3"] = fdom.size
    h["NAXIS4"] = pdom.size
    h["DATE-MAP"] = Time(time.time(), format="unix").iso.split()[0]
    if direction is not None:
        h["EQUINOX"] = direction.equinox

    for kk, vv in header_override.items():
        h[kk] = vv

    for t_ind, t_val in enumerate(tdom.coordinates):
        val = field.val[:, t_ind]  # Select time
        val = np.transpose(val, (0, 1, 3, 2))  # Switch spatial axes
        val = val.astype(np.float32)
        val = np.require(val, dtype=np.float32, requirements="F")
        hdu = pyfits.PrimaryHDU(val, header=h)
        hdulist = pyfits.HDUList([hdu])
        base, ext = splitext(file_name)
        hdulist.writeto(base + f"time{t_val}" + ext, overwrite=True)
