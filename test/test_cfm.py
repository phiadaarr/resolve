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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import nifty8 as ift

import resolve as rve
from .common import setup_function, teardown_function
import pytest

pmp = pytest.mark.parametrize


@pmp("total_N", [1, 2])
@pmp("prefix", ["", "b"])
@pmp("nthreads", [1, 2])
@pmp("cfg", list(range(7)))
@pmp("dofdex_mode", [0, 1])
def test_cfm(total_N, prefix, cfg, nthreads, dofdex_mode):
    ift.set_nthreads(nthreads)

    dom1 = dom2 = None
    if cfg == 0:
        dom0 = ift.RGSpace(20)
        dom1 = ift.RGSpace(22)
    elif cfg == 1:
        dom0 = ift.RGSpace(100, 1000)
        dom1 = ift.RGSpace(200, 1.2)
    elif cfg == 2:
        dom0 = ift.RGSpace(4, 1.231)
        dom1 = ift.RGSpace((30, 30), (0.1, 0.1))
    elif cfg == 3:
        dom0 = ift.RGSpace((12, 12), (0.1, 0.1))
        dom1 = ift.RGSpace(20, 1.231)
    elif cfg == 4:
        dom0 = ift.RGSpace((12, 12), (0.1, 0.1))
    elif cfg == 5:
        dom0 = ift.RGSpace(20, 1.231)
    elif cfg == 6:
        dom0 = ift.RGSpace((12, 12), (0.1, 0.1))
        dom1 = ift.RGSpace(20, 1.231)
        dom2 = ift.RGSpace(22)

    if dofdex_mode == 0:
        dofdex = list(range(total_N))
    else:
        dofdex = total_N * [0]

    args_cfm = dict(prefix=prefix, total_N=total_N)
    args0 = dict(
        target_subdomain=dom0,
        fluctuations=(1.0, 1.0),
        flexibility=(2.0, 2),
        asperity=(0.1, 0.1),
        loglogavgslope=(-2, 0.1),
        prefix="dom0",
        dofdex=dofdex,
    )
    args1 = dict(
        target_subdomain=dom1,
        fluctuations=(2.0, 0.1),
        flexibility=(1.0, 2),
        asperity=(0.2, 0.1),
        loglogavgslope=(-3, 0.321),
        prefix="dom1",
        dofdex=dofdex,
    )
    args2 = dict(
        target_subdomain=dom2,
        fluctuations=(2.0, 0.1),
        flexibility=(1.0, 2),
        asperity=(0.2, 0.1),
        loglogavgslope=(-3, 0.321),
        prefix="dom2",
        dofdex=dofdex,
    )
    args_zm = dict(offset_mean=1.12, offset_std=(1.0, 0.2), dofdex=dofdex)
    cfm0 = ift.CorrelatedFieldMaker(**args_cfm)
    cfm = rve.CorrelatedFieldMaker(**args_cfm, nthreads=nthreads)
    cfm0.add_fluctuations(**args0)
    cfm.add_fluctuations(**args0)
    if dom1 is not None:
        cfm0.add_fluctuations(**args1)
        cfm.add_fluctuations(**args1)
        if dom2 is not None:
            cfm0.add_fluctuations(**args2)
            cfm.add_fluctuations(**args2)
    cfm0.set_amplitude_total_offset(**args_zm)
    cfm.set_amplitude_total_offset(**args_zm)

    op0 = cfm0.finalize(0)
    op1 = cfm.finalize(0)
    op2 = cfm.finalize_nifty_equivalent(0)

    # TEMPORARY
    pos = ift.from_random(op0.domain)
    ift.extra.assert_allclose(op0(pos), op1(pos), rtol=1e-5)
    ift.extra.assert_allclose(op0(pos), op2(pos), rtol=1e-5)

    pos = ift.Linearization.make_var(ift.from_random(op0.domain))
    ift.extra.assert_allclose(op0(pos).val, op1(pos).val, rtol=1e-5)
    ift.extra.assert_allclose(op0(pos).val, op2(pos).val, rtol=1e-5)

    pos = ift.Linearization.make_var(ift.from_random(op0.domain))
    pos1 = ift.from_random(op0.domain)
    ift.extra.assert_allclose(op0(pos).jac(pos1), op1(pos).jac(pos1), rtol=1e-5)
    ift.extra.assert_allclose(op0(pos).jac(pos1), op2(pos).jac(pos1), rtol=1e-5)

    pos = ift.Linearization.make_var(ift.from_random(op0.domain))
    pos1 = ift.from_random(op0.target)

    ift.extra.assert_allclose(
        op0(pos).jac.adjoint(pos1), op1(pos).jac.adjoint(pos1), rtol=1e-5
    )
    ift.extra.assert_allclose(
        op0(pos).jac.adjoint(pos1), op2(pos).jac.adjoint(pos1), rtol=1e-5
    )

    return
    # /TEMPORARY

    rve.operator_equality(
        op0, op1, rtol=1e-5, ntries=3
    )  # FIXME Why is the accuracy so low?
    rve.operator_equality(op0, op2, rtol=1e-5, ntries=3)
