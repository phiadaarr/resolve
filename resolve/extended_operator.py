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


class ExtendedOperator(ift.Operator):
    @property
    def _domain(self):
        return self._op.domain

    @property
    def _target(self):
        return self._op.target

    def apply(self, x):
        self._check_input(x)
        return self._op(x)

    def __repr__(self):
        return self._op.__repr__()


class AXiOperator(ExtendedOperator):
    def adjust_variances(self, position, minimizer, samples):
        # TODO Use do_adjust_variances from nifty?
        # Check whether necessary stuff is defined
        self._A, self._xi, self._op

        ham = ift.make_adjust_variances_hamiltonian(
            self._A, self._xi, position, samples=samples)

        # Minimize
        e = ift.EnergyAdapter(
            position.extract(self._A.domain),
            ham,
            constants=[],
            want_metric=True)
        e, _ = minimizer(e)

        # Temp
        oldres = self._op.force(position).to_global_data()

        # Update position
        s_h_old = (self._A*self._xi).force(position)
        position = position.to_dict()
        position[list(self._xi.domain.keys())[0]] = s_h_old/self._A(e.position)
        position = ift.MultiField.from_dict(position)
        position = ift.MultiField.union([position, e.position])

        # TODO Move this into the tests
        import numpy as np
        newres = self._op.force(position).to_global_data()
        np.testing.assert_allclose(oldres, newres)
        return position
