# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift


class DtypeConverter(ift.EndomorphicOperator):
    def __init__(self, domain, domain_dtype, target_dtype, hint="", casting="same_kind"):
        self._domain = ift.DomainTuple.make(domain)
        self._ddt = domain_dtype
        self._tdt = target_dtype
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._hint = hint
        self._casting = casting

    def apply(self, x, mode):
        self._check_input(x, mode)
        # Sanity check
        if mode == self.TIMES:
            inp, out = self._ddt, self._tdt
        else:
            out, inp = self._ddt, self._tdt
        if inp is not None and x.dtype != inp:
            s = ["Dtypes not compatible", str(self.domain),
                 f"Input: {x.dtype}, should be: {inp}", self._hint]
            raise ValueError("\n".join(s))
        # /Sanity check
        if inp is None:
            return x
        return ift.makeField(self._tgt(mode),
                             x.val.astype(out, casting=self._casting, copy=False))
