# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras, Jakob Knollmüller

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
        return f"PolarizationSpace(polarization_labels={self._lbl})"

    @property
    def labels(self):
        return self._lbl
