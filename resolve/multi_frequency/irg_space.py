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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

import nifty7 as ift


class IRGSpace(ift.StructuredDomain):
    """Represents non-equidistantly binned and non-periodic one-dimensional spaces.

    Parameters
    ----------
    binbounds : np.ndarray
        Must be sorted and strictly ascending.
    """

    _needed_for_hash = ["_binbounds"]

    def __init__(self, binbounds):
        bb = np.array(binbounds)
        if bb.ndim != 1:
            raise TypeError
        if np.any(np.diff(bb) <= 0.):
            raise ValueError("Binbounds must be sorted and strictly ascending")
        self._binbounds = tuple(bb)

    def __repr__(self):
        return (f"IRGSpace(shape={self.shape}, binbounds=...)")

    @property
    def harmonic(self):
        """bool : Always False for this class."""
        return False

    @property
    def shape(self):
        return (len(self._binbounds) - 1,)

    @property
    def size(self):
        return self.shape[0]

    @property
    def scalar_dvol(self):
        return None

    @property
    def dvol(self):
        # FIXME Is this volume treatment really correct?
        return np.diff(np.array(self._binbounds))[:-1]

    @property
    def binbounds(self):
        return self._binbounds

    # @property
    # def pixel_centers(self):
    #     bb = np.array(self._binbounds)
    #     return bb[:-1] + np.diff(bb)/2.
