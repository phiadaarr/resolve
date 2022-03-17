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
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np

from .util import my_assert, my_assert_isinstance, my_asserteq


def MultiDomainVariableCovarianceGaussianEnergy(data, signal_response, invcov):
    from .likelihood import get_mask_multi_field

    my_asserteq(data.domain, signal_response.target, invcov.target)
    my_assert_isinstance(data.domain, ift.MultiDomain)
    my_assert_isinstance(signal_response.domain, ift.MultiDomain)
    my_assert(ift.is_operator(invcov))
    my_assert(ift.is_operator(signal_response))
    res = []
    invcovfld = invcov(ift.full(invcov.domain, 1.0))
    mask = get_mask_multi_field(invcovfld)
    data = mask(data)
    signal_response = mask @ signal_response
    invcov = mask @ invcov
    for kk in data.keys():
        res.append(
            ift.VariableCovarianceGaussianEnergy(
                data.domain[kk], "resi" + kk, "icov" + kk, data[kk].dtype
            )
        )
    resi = ift.PrependKey(data.domain, "resi") @ ift.Adder(data, True) @ signal_response
    invcov = ift.PrependKey(data.domain, "icov") @ invcov
    return reduce(add, res) @ (resi + invcov)


class MultiFieldStacker(ift.LinearOperator):
    def __init__(self, target, space, keys):
        """
        Parameters
        ----------
        target : DomainTuple
            Target of the operator.
        space : int
            Index behind which the iterated domain shall be in the target
            domain. target[space] needs to be one-dimensional.
        keys : list
            List of keys. Needs to have same length as target[space].
        """
        self._target = ift.DomainTuple.make(target)
        dom = {kk: self._target[:space] + self._target[space + 1:] for kk in keys}
        self._domain = ift.makeDomain(dom)
        self._capability = self._all_ops
        self._keys = tuple(keys)
        self._space = int(space)
        assert tuple([len(self._keys)]) == target[self._space].shape
        assert self.domain.size == self.target.size

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES or mode == self.ADJOINT_INVERSE_TIMES:
            dtype = x[self._keys[0]].dtype
            for kk in self._keys:
                assert x[kk].dtype == dtype
            res = np.empty(self._target.shape, dtype=dtype)
            for ii, kk in enumerate(self._keys):
                res[self._slice(ii)] = x[kk]
        else:
            res = {kk: x[self._slice(ii)] for ii, kk in enumerate(self._keys)}
        return ift.makeField(self._tgt(mode), res)

    def _slice(self, index):
        ndim = reduce(add, (len(self._target[ii].shape) for ii in range(self._space)), 0)
        return ndim * (slice(None),) + (index,)
