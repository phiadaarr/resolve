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
    coordinates : np.ndarray
        Must be sorted and strictly ascending.
    """

    _needed_for_hash = ["_coordinates"]

    def __init__(self, coordinates):
        bb = np.array(coordinates)
        if bb.ndim != 1:
            raise TypeError
        if np.any(np.diff(bb) <= 0.):
            raise ValueError("Coordinates must be sorted and strictly ascending")
        self._coordinates = tuple(bb)

    def __repr__(self):
        return (f"IRGSpace(shape={self.shape}, coordinates=...)")

    @property
    def harmonic(self):
        """bool : Always False for this class."""
        return False

    @property
    def shape(self):
        return len(self._coordinates),

    @property
    def size(self):
        return self.shape[0]

    @property
    def scalar_dvol(self):
        return None

    @property
    def dvol(self):
        """This has one pixel less than :meth:`shape`. Not sure if this is
        okay.
        """
        return np.diff(np.array(self._coordinates))

    @property
    def coordinates(self):
        return self._coordinates