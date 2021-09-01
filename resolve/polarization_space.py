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
# Copyright(C) 2013-2021 Max-Planck-Society

import nifty8 as ift

from .util import my_assert


class PolarizationSpace(ift.UnstructuredDomain):
    """

    Parameters
    ----------
    coordinates : np.ndarray
        Must be sorted and strictly ascending.
    """

    _needed_for_hash = ["_shape", "_lbl"]

    def __init__(self, polarization_labels):
        if isinstance(polarization_labels, str):
            polarization_labels = (polarization_labels,)
        self._lbl = tuple(polarization_labels)
        for ll in self._lbl:
            my_assert(ll in ["I", "Q", "U", "V", "LL", "LR", "RL", "RR", "XX", "XY", "YX", "YY"])
        super(PolarizationSpace, self).__init__(len(self._lbl))

    def __repr__(self):
        return f"PolarizationSpace({self._lbl})"

    @property
    def labels(self):
        return self._lbl
