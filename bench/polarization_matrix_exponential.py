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
# Author: Philipp Arras

import sys

import resolve as rve
import nifty8 as ift
import numpy as np

from time import time

if len(sys.argv) == 2 and sys.argv[1] == "quick":
    npix = 10
else:
    npix = 4000

pdom = rve.PolarizationSpace(["I", "Q", "U", "V"])
sdom = ift.RGSpace([npix, npix])
dom = rve.default_sky_domain(pdom=pdom, sdom=sdom)
dom = {kk: dom[1:] for kk in pdom.labels}
tgt = rve.default_sky_domain(pdom=pdom, sdom=sdom)
opold = rve.polarization_matrix_exponential(tgt)
opold_jax = rve.polarization_matrix_exponential(tgt, jax=True)


if False:
    op = rve.cpp.PolarizationMatrixExponential(dom, 8)
    loc = ift.from_random(op.domain)
    for ii in range(10000):
        print(ii)
        op(ift.Linearization.make_var(loc))

for nthreads in [1, 4, 8]:
    op = rve.polarization_matrix_exponential_mf2f(dom, nthreads)
    print(f"New implementation (nthreads={nthreads})")
    ift.exec_time(op)
    print()
print("Old implementation")
ift.exec_time(opold)
print()
print("Old implementation (jax)")
ift.exec_time(opold_jax)
