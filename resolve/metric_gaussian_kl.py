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
# Copyright(C) 2019 Max-Planck-Society

import nifty5 as ift

from .key_handler import key_handler


def MetricGaussianKL(position,
                     ham,
                     nsamp,
                     constants=[],
                     point_estimates=key_handler.zeromodes):
    kl = ift.MetricGaussianKL(
        position,
        ham,
        nsamp,
        constants=constants,
        point_estimates=point_estimates,
        mirror_samples=True)
    sc = ift.StatCalculator()
    for samp in kl.samples:
        sc.add(ham(kl.position.flexible_addsub(samp, False)))
    print('Hamiltonian = {:.2E} +/- {:.2E}'.format(
        float(sc.mean.to_global_data()),
        float(ift.sqrt(sc.var).to_global_data())))
    return kl
