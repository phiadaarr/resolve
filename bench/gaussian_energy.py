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

import sys
from time import time

import nifty8 as ift
import numpy as np

import resolve as rve

if len(sys.argv) == 2 and sys.argv[1] == "quick":
    shp = (10,)
else:
    shp = (1000000, 100)  # 100 Mio entries

dom = ift.UnstructuredDomain(shp)
mean = ift.full(dom, 1.2)
invcov = ift.full(dom, 142.1)

print("Gaussian energy")
print("^^^^^^^^^^^^^^^")
for nthreads in [1, 4, 8]:
    op = rve.DiagonalGaussianLikelihood(mean, invcov, nthreads=nthreads)
    print(f"New implementation (nthreads={nthreads})")
    ift.exec_time(op)
    print()
print("Old implementation")
opold = ift.GaussianEnergy(mean, ift.makeOp(invcov))
ift.exec_time(opold)
print()

print("Variable covariance Gaussian energy")
print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
for nthreads in [1, 4, 8]:
    op = rve.VariableCovarianceDiagonalGaussianLikelihood(mean, "signal", "logicov",
                                                          nthreads=nthreads)
    print(f"New implementation (nthreads={nthreads})")
    ift.exec_time(op)
    print()

print("Old implementation")
ift.exec_time(op.nifty_equivalent)
