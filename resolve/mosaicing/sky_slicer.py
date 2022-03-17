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
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np


class SkySlicer(ift.LinearOperator):
    """Maps from the total sky domain to individual sky domains and applies the
    primary beam pattern.

    Parameters
    ----------
    domain : RGSpace
        Two-dimensional RG-Space which serves as domain. The distances are
        in pseudo-radian.

    beam_directions : dict(key: BeamDirection)
        Dictionary of BeamDirection that contains the information of the
        directions of the different observations and the beam pattern.
    """

    def __init__(self, domain, beam_directions):
        self._bd = dict(beam_directions)
        self._domain = ift.makeDomain(domain)
        t, b, s = {}, {}, {}
        for kk, vv in self._bd.items():
            print("\r" + kk, end="")
            t[kk], s[kk], b[kk] = vv.slice_target(self._domain)
        print()
        self._beams = b
        self._slices = s
        self._target = ift.makeDomain(t)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = {}
            for kk in self._target.keys():
                res[kk] = x[self._slices[kk]] * self._beams[kk].val
        else:
            res = np.zeros(self._domain.shape)
            for kk, xx in x.items():
                res[self._slices[kk]] += xx * self._beams[kk].val
        return ift.makeField(self._tgt(mode), res)


class BeamDirection:
    """Represent direction information of one pointing.

    Parameters
    ----------
    dx : float
        Pointing offset in pseudo radian.
    dy : float
        Pointing offset in pseudo radian.
    beam_func : function
        Function that takes the two directions on the sky in pseudo radian and
        returns beam pattern.
    cutoff : float
        Relative area of beam pattern that is cut off. 0 corresponds to no
        cutoff at all. 1 corresponds to cut off everything.
    """

    def __init__(self, dx, dy, beam_func, cutoff):
        self._dx, self._dy = float(dx), float(dy)
        self._f = beam_func
        self._cutoff = float(cutoff)
        assert 0. <= cutoff < 1

    def slice_target(self, domain):
        """
        Parameters
        ----------
        domain : RGSpace
            Total sky domain.
        """
        dom = ift.makeDomain(domain)[0]
        dst = np.array(dom.distances)
        assert abs(self._dx) < dst[0] * dom.shape[0]
        assert abs(self._dy) < dst[1] * dom.shape[1]
        npix = np.array(domain.shape)

        xs = np.linspace(0, max(npix * dst), num=4*max(npix))
        # Technically not 100% correct since we integrate circles here.
        # The actual cutoff is lower than the one that is induced by the
        # following computation.
        ys = self._f(xs)*xs
        cond = np.cumsum(ys)/np.sum(ys)
        np.testing.assert_allclose(cond[-1], 1.)
        ind = np.searchsorted(cond, 1.-self._cutoff)
        fov = 2*xs[ind]
        patch_npix = fov/dst
        patch_npix = (np.ceil(patch_npix/2)*2).astype(int)

        # Convention: pointing convention (see also SingleDishResponse)
        shift = np.array([-self._dx, self._dy])
        patch_center_unrounded = np.array(npix) / 2 + shift / dst
        ipix_patch_center = np.round(patch_center_unrounded).astype(int)
        # Or maybe use ceil?
        assert np.all(patch_npix % 2 == 0)
        mi = (ipix_patch_center - patch_npix / 2).astype(int)
        ma = mi + patch_npix

        assert np.all(mi >= 0)
        assert np.all(ma < npix)
        slc = slice(mi[0], ma[0]), slice(mi[1], ma[1])
        tgt = ift.RGSpace(patch_npix, dst)
        assert tgt.shape[0] % 2 == 0
        assert tgt.shape[1] % 2 == 0
        test = np.empty(dom.shape)
        assert test[slc].shape == tgt.shape

        # Create coordinate field
        mgrid = np.mgrid[: patch_npix[0], : patch_npix[1]].astype(float)
        mgrid -= patch_npix[..., None, None] / 2
        # FIXME Add subpixel offset
        # subpixel_offset = ipix_patch_center - patch_center_unrounded
        # assert np.all(subpixel_offset < 1.0)
        # assert np.all(subpixel_offset >= 0.0)
        mgrid *= dst[..., None, None]
        beam = ift.makeField(tgt, self._f(np.linalg.norm(mgrid, axis=0)))
        return tgt, slc, beam
