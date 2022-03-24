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
import sys
import pytest

pmp = pytest.mark.parametrize

nthreads = 1


if len(sys.argv) == 2 and sys.argv[1] == "quick":
    total_N = 2
    dom0 = ift.RGSpace(4)
    dom1 = ift.RGSpace(5)
else:
    total_N = 60
    dom0 = ift.RGSpace(1000)
    dom1 = ift.RGSpace(100)


dofdex = list(range(total_N))
args0 = dict(prefix="", total_N=total_N)
args1 = dict(
    target_subdomain=dom0,
    fluctuations=(1.0, 1.0),
    flexibility=(2.0, 2),
    asperity=(0.1, 0.1),
    loglogavgslope=(-2, 0.1),
    prefix="dom0",
    dofdex=dofdex,
)
args2 = dict(
    target_subdomain=dom1,
    fluctuations=(2.0, 0.1),
    flexibility=(1.0, 2),
    asperity=(0.2, 0.1),
    loglogavgslope=(-3, 0.321),
    prefix="dom1",
    dofdex=dofdex,
)
args3 = dict(offset_mean=1.2, offset_std=(1.0, 0.2), dofdex=dofdex)
cfm = ift.CorrelatedFieldMaker(**args0)
cfm.add_fluctuations(**args1)
cfm.add_fluctuations(**args2)
cfm.set_amplitude_total_offset(**args3)
op0 = cfm.finalize(0)
ift.set_nthreads(nthreads)

cfm = rve.CorrelatedFieldMaker(**args0, nthreads=nthreads)
cfm.add_fluctuations(**args1)
cfm.add_fluctuations(**args2)
cfm.set_amplitude_total_offset(**args3)
op1 = cfm.finalize()

pos = ift.from_random(op0.domain)

print("New implementation")
try:
    ift.exec_time(op0)
except:
    pass

print("Old implementation")
try:
    ift.exec_time(op1)
except:
    pass
