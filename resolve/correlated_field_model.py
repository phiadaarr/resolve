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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

from functools import reduce
from operator import add, mul

import nifty8 as ift
import numpy as np
import resolve_support

from .cpp2py import Pybind11Operator


class CorrelatedFieldMaker(ift.CorrelatedFieldMaker):
    def __init__(self, prefix, total_N=1, nthreads=1):
        if total_N <= 0:
            raise ValueError("total_N must be > 0.")
        self._nthreads = nthreads
        super(CorrelatedFieldMaker, self).__init__(prefix, total_N)

    def add_fluctuations(self, *args, **kwargs):
        if "dofdex" not in kwargs:
            raise RuntimeError("Need to specify dofdex")
        super(CorrelatedFieldMaker, self).add_fluctuations(*args, **kwargs)

    def set_amplitude_total_offset(self, *args, **kwargs):
        if "dofdex" not in kwargs:
            raise RuntimeError("Need to specify dofdex")
        super(CorrelatedFieldMaker, self).set_amplitude_total_offset(*args, **kwargs)

    def _finalize(self, prior_info=100):
        n_amplitudes = len(self._a)
        hspace = ift.makeDomain(
            [ift.UnstructuredDomain(self._total_N)]
            + [dd.target[-1].harmonic_partner for dd in self._a]
        )
        spaces = tuple(range(1, n_amplitudes + 1))
        amp_space = 1

        a = list(self.get_normalized_amplitudes())
        pspaces = [aa.target for aa in a]
        power_keys = [str(ii) for ii in range(len(a))]

        core = CfmCore(
            pspaces,
            power_keys,
            self._prefix + "xi",
            self._total_N,
            self._target_subdomains,
            self._offset_mean,
            self.azm,
            self._nthreads,
        )

        amplitudes = reduce(
            add,
            [
                aa.ducktape_left(kk)
                for kk, aa in zip(power_keys, self.get_normalized_amplitudes())
            ],
        )
        op = core.partial_insert(amplitudes)
        self.statistics_summary(prior_info)
        return op, core.nifty_equivalent.partial_insert(amplitudes)

    def finalize(self, prior_info):
        return self._finalize(prior_info)[0]

    def finalize_nifty_equivalent(self, prior_info):
        return self._finalize(prior_info)[1]


def CfmCore(
    pdomains,
    power_keys,
    excitation_field_key,
    total_N,
    target_subdomains,
    offset_mean,
    azm,
    nthreads=1,
):
    hspace = ift.makeDomain(
        [ift.UnstructuredDomain(total_N)] + [dd[-1].harmonic_partner for dd in pdomains]
    )
    n_amplitudes = len(pdomains)
    assert len(pdomains) == len(power_keys)
    spaces = tuple(range(1, n_amplitudes + 1))
    amp_space = 1

    ht = ift.HarmonicTransformOperator(
        hspace, target_subdomains[0][amp_space], space=spaces[0]
    )
    for i in range(1, n_amplitudes):
        ht = (
            ift.HarmonicTransformOperator(
                ht.target, target_subdomains[i][amp_space], space=spaces[i]
            )
            @ ht
        )
    a = []
    for ii in range(n_amplitudes):
        co = ift.ContractionOperator(hspace, spaces[:ii] + spaces[ii + 1 :])
        pp = pdomains[ii][amp_space]
        pd = ift.PowerDistributor(co.target, pp, amp_space)
        a.append(
            co.adjoint
            @ pd
            @ ift.Operator.identity_operator(pd.domain).ducktape(power_keys[ii])
        )
    corr = reduce(mul, a)
    xi = ift.ducktape(hspace, None, excitation_field_key)

    from nifty8.library.correlated_fields import _Distributor
    expander = ift.ContractionOperator(hspace, spaces=spaces).adjoint
    dofdex = range(total_N)
    distributor = _Distributor(dofdex, azm.target, ift.UnstructuredDomain(total_N))

    azm_key = azm.domain.keys()[0]

    op = ht((expander @ ift.Operator.identity_operator(distributor.target).ducktape(azm_key)) * corr * xi)

    if offset_mean is None:
        offset_mean = 0.
    else:
        if not isinstance(offset_mean, float):
            raise NotImplementedError(f"offset_mean needs to be a float, got {offset_mean}")
        op = ift.Adder(ift.full(op.target, float(offset_mean))) @ op

    pindices = [pp[amp_space].pindex for pp in pdomains]

    scalar_dvol = ht.domain.scalar_weight(spaces)
    offset_mean /= scalar_dvol

    res = Pybind11Operator(
        op.domain,
        op.target,
        resolve_support.CfmCore(pindices, power_keys, excitation_field_key, azm_key, offset_mean, scalar_dvol, nthreads),
    )
    azm_insert = (distributor @ azm).ducktape_left(azm_key)
    res = res.partial_insert(azm_insert)
    res.nifty_equivalent = op.partial_insert(azm_insert)
    return res
