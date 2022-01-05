# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras, Jakob Knollmüller

import nifty8 as ift
import numpy as np

from .util import my_assert


class PolarizationSpace(ift.UnstructuredDomain):
    """

    Parameters
    ----------
    coordinates : np.ndarray
        Must be sorted and strictly ascending.
    """

    _needed_for_hash = ["_hash_lbl"]
    _allowed = ["I", "Q", "U", "V", "LL", "LR", "RL", "RR", "XX", "XY", "YX", "YY"]

    def __init__(self, polarization_labels):
        if isinstance(polarization_labels, str):
            polarization_labels = [polarization_labels]
        self._lbl = tuple(polarization_labels)
        for ll in self._lbl:
            my_assert(ll in PolarizationSpace._allowed)
        super(PolarizationSpace, self).__init__(len(self._lbl))
        # Note: hash of string is not reproducible accross runs
        self._hash_lbl = tuple(PolarizationSpace._allowed.index(ll) for ll in self._lbl)

    def __repr__(self):
        return f"PolarizationSpace(polarization_labels={self._lbl})"

    @property
    def labels(self):
        return self._lbl

    def labels_eq(self, lst):
        return set(lst) == set(self._lbl)

    def label2index(self, label):
        return self._lbl.index(label)


def polarization_converter(domain, target):
    from .util import my_assert_isinstance

    domain = ift.DomainTuple.make(domain)
    target = ift.DomainTuple.make(target)
    my_assert_isinstance(domain[0], PolarizationSpace)
    my_assert_isinstance(target[0], PolarizationSpace)
    if domain is target:
        return ift.ScalingOperator(domain, 1.)

    if domain[0].labels_eq("I"):
        if target[0].labels in [("LL", "RR"), ("XX", "YY")]:
            # Convention: Stokes I 1Jy source leads to 1Jy in LL and 1Jy in RR
            op = ift.ContractionOperator(target, 0).adjoint
            return op.ducktape(domain)
    if domain[0].labels_eq(["I", "Q", "U"]):
        if target[0].labels_eq(["LL", "RR", "LR", "RL"]):
            op = _PolarizationConverter(domain, target, 0)
            #ift.extra.check_linear_operator(op, complex, complex)
            return op
    raise NotImplementedError(f"Polarization converter\ndomain:\n{domain[0]}\ntarget\n{target[0]}\n")


class _PolarizationConverter(ift.LinearOperator):
    def __init__(self, domain, target, space):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._space = int(space)

        assert self._space < len(self._domain)
        for ii in range(len(self._domain)):
            if ii == self._space:
                assert isinstance(self._domain[space], PolarizationSpace)
                assert isinstance(self._target[space], PolarizationSpace)
            else:
                assert self._domain[ii] == self._target[ii]

    def apply(self, x, mode):
        self._check_input(x, mode)
        polx = lambda lbl: x.val[x.domain[self._space].label2index(lbl)]
        f = lambda s: self._tgt(mode)[self._space].label2index(s)
        res = np.empty(self._tgt(mode).shape, dtype=x.dtype)
        if mode == self.TIMES:
            # Convention: Stokes I 1Jy source leads to 1Jy in LL and 1Jy in RR
            res[f("LL")] = res[f("RR")] = polx("I")
            # if with_v:
            #     LL += V
            #     RR -= -V
            res[f("RL")] = polx("Q")+1j*polx("U")
            res[f("LR")] = polx("Q")-1j*polx("U")
        else:
            res[f("I")] = polx("LL") + polx("RR")
            # if with_v:
            #     V = LL - RR
            res[f("Q")] = polx("LR") + polx("RL")
            res[f("U")] = 1j*(polx("LR") - polx("RL"))
        return ift.makeField(self._tgt(mode), res)
