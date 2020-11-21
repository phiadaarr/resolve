# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import time
from os.path import splitext

import numpy as np

from .mpi import onlymaster


@onlymaster
def field2fits(field, file_name, overwrite, direction=None):
    from astropy.time import Time
    import astropy.io.fits as pyfits

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

    # @staticmethod
    # def make_from_file(file_name):
    #     with pyfits.open(file_name) as hdu_list:
    #         lst = hdu_list[0]
    #         pcx = lst.header['CRVAL1']/180*np.pi
    #         pcy = lst.header['CRVAL2']/180*np.pi
    #         equ = lst.header['EQUINOX']
    #     return FitsWriter([pcx, pcy], equ)

    # @staticmethod
    # def fits2field(file_name, ignore_units=False, from_wsclean=False):
    #     with pyfits.open(file_name) as hdu_list:
    #         image_data = np.squeeze(hdu_list[0].data).astype(np.float64)
    #         head = hdu_list[0].header
    #         dstx = abs(head['CDELT1']*np.pi/180)
    #         dsty = abs(head['CDELT2']*np.pi/180)
    #         if not ignore_units:
    #             if head['BUNIT'] == 'JY/BEAM':
    #                 fac = np.pi/4/np.log(2)
    #                 scale = fac*head['BMAJ']*head['BMIN']*(np.pi/180)**2
    #             elif head['BUNIT'] == 'JY/PIXEL':
    #                 scale = dstx*dsty
    #             else:
    #                 scale = 1
    #             image_data /= scale
    #     if from_wsclean:
    #         image_data = image_data[::-1].T[:, :-1]
    #         image_data = np.pad(image_data, ((0, 0), (1, 0)), mode='constant')
    #     else:
    #         image_data = image_data.T[:, ::-1]
    #     dom = ift.RGSpace(image_data.shape, (dstx, dsty))
    #     return ift.makeField(dom, image_data)
