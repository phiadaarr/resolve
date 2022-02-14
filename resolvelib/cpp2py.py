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
# Copyright(C) 2021-2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras, Jakob Roth

import numpy as np
import nifty8 as ift


class Pybind11Operator(ift.Operator):
    def __init__(self, dom, tgt, op):
        self._domain = ift.makeDomain(dom)
        self._target = ift.makeDomain(tgt)
        self._op = op

    def apply(self, x):
        self._check_input(x)
        if ift.is_linearization(x):
            lin = self._op.apply_with_jac(x.val.val)
            jac = Pybind11Jacobian(self._domain, self._target, lin.jac_times, lin.jac_adjoint_times)
            pos = ift.makeField(self._target, lin.position())
            return x.new(pos, jac)
        return ift.makeField(self.target, self._op.apply(x.val))


class Pybind11Jacobian(ift.LinearOperator):
    def __init__(self, domain, target, times, adj_times):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._times = times
        self._adj_times = adj_times

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = (self._times if mode == self.TIMES else self._adj_times)(x.val)
        return ift.makeField(self._tgt(mode), res)
