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
