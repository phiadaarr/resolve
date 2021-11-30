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
            polarization_labels = [polarization_labels]
        lbl = list(polarization_labels)
        lbl.sort()
        self._lbl = tuple(polarization_labels)
        for ll in self._lbl:
            my_assert(ll in ["I", "Q", "U", "V", "LL", "LR", "RL", "RR", "XX", "XY", "YX", "YY"])
        super(PolarizationSpace, self).__init__(len(self._lbl))

    def __repr__(self):
        return f"PolarizationSpace(polarization_labels={self._lbl})"

    @property
    def labels(self):
        return self._lbl


def polarization_converter(domain, target):
    from .util import my_assert_isinstance
    from .simple_operators import DomainChangerAndReshaper

    domain = ift.DomainTuple.make(domain)
    target = ift.DomainTuple.make(target)
    my_assert_isinstance(domain[0], PolarizationSpace)
    my_assert_isinstance(target[0], PolarizationSpace)
    if domain is target:
        return ift.ScalingOperator(domain, 1.)

    if domain[0].labels == ("I",):
        if target[0].labels in [("LL", "RR"), ("XX", "YY")]:
            # Convention: Stokes I 1Jy source leads to 1Jy in LL and 1Jy in RR
            op = ift.ContractionOperator(target, 0).adjoint
            return op @ DomainChangerAndReshaper(domain, op.domain)
    raise NotImplementedError(f"Polarization converter\ndomain:\n{domain[0]}\ntarget\n{target[0]}\n")
