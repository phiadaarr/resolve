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

import time
from os.path import splitext

import nifty8 as ift
import numpy as np

from .constants import DEG2RAD
from .mpi import onlymaster


@onlymaster
def field2fits(field, file_name, overwrite, direction=None):
    import astropy.io.fits as pyfits
    from astropy.time import Time

    dom0 = field.domain
    assert len(dom0) == 1
    assert len(dom0[0].shape) == 2
    if direction is not None:
        pcx, pcy = direction.phase_center
    dom = dom0[0]
    h = pyfits.Header()
    h["BUNIT"] = "Jy/sr"
    h["CTYPE1"] = "RA---SIN"
    h["CRVAL1"] = pcx * 180 / np.pi if direction is not None else 0.0
    h["CDELT1"] = -dom.distances[0] * 180 / np.pi
    h["CRPIX1"] = dom.shape[0] / 2
    h["CUNIT1"] = "deg"
    h["CTYPE2"] = "DEC---SIN"
    h["CRVAL2"] = pcy * 180 / np.pi if direction is not None else 0.0
    h["CDELT2"] = dom.distances[1] * 180 / np.pi
    h["CRPIX2"] = dom.shape[1] / 2
    h["CUNIT2"] = "deg"
    h["DATE-MAP"] = Time(time.time(), format="unix").iso.split()[0]
    if direction is not None:
        h["EQUINOX"] = direction.equinox
    hdu = pyfits.PrimaryHDU(field.val[:, :].T, header=h)
    hdulist = pyfits.HDUList([hdu])
    base, ext = splitext(file_name)
    hdulist.writeto(base + ext, overwrite=overwrite)


def fits2field(file_name, ignore_units=False, from_wsclean=False):
    import astropy.io.fits as pyfits

    with pyfits.open(file_name) as hdu_list:
        image_data = hdu_list[0].data.astype(np.float64, casting="same_kind", copy=False)
        assert image_data.shape[0] == 1  # Only one Stokes component
        image_data = image_data[0]
        head = hdu_list[0].header
        assert head["CUNIT1"].strip() == "deg"
        assert head["CUNIT2"].strip() == "deg"
        assert head["CUNIT3"].strip() == "Hz"
        refs = []
        refs.append([float(head["CRVAL3"]), int(head["CRPIX3"]), head["CDELT3"]])
        refs.append(
            [
                float(head["CRVAL2"]) * DEG2RAD,
                int(head["CRPIX2"]),
                head["CDELT2"] * DEG2RAD,
            ]
        )
        refs.append(
            [
                float(head["CRVAL1"]) * DEG2RAD,
                int(head["CRPIX1"]),
                head["CDELT1"] * DEG2RAD,
            ]
        )

        if not ignore_units:
            if head["BUNIT"].upper() == "JY/BEAM":
                fac = np.pi / 4 / np.log(2)
                scale = fac * head["BMAJ"] * head["BMIN"] * (np.pi / 180) ** 2
            elif head["BUNIT"].upper() == "JY/PIXEL":
                scale = abs(refs[0][2] * refs[1][2])
            else:
                scale = 1
            image_data /= scale

    # Convert CASA conventions to resolve conventions
    inds = 0, 2, 1
    image_data = np.transpose(image_data, inds)
    refs = [refs[ii] for ii in inds]
    refs[1][2] *= -1

    for ii, (_, mypx, mydst) in enumerate(refs):
        if mydst == 0.0:
            raise RuntimeError
        if mydst > 0:
            continue
        image_data = np.flip(image_data, ii)
        refs[ii][2] *= -1
        # FIXME Assume pixel counting start at 0. Maybe also 1?
        refs[ii][1] = image_data.shape[ii] - mypx

    refval = tuple(refs[ii][0] for ii in range(3))
    refpx = tuple(refs[ii][1] for ii in range(3))
    dsts = tuple(refs[ii][2] for ii in range(3))
    dom = (
        ift.RGSpace(image_data.shape[0], dsts[0]),
        ift.RGSpace(image_data.shape[1:], dsts[1:]),
    )
    refval = tuple(refval[ii] - refpx[ii] * dsts[ii] for ii in range(3))
    del refpx
    return ift.makeField(dom, image_data), refval
