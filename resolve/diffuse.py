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

from .extended_operator import AXiOperator
from .key_handler import key_handler


class Diffuse(AXiOperator):
    def __init__(self, target, amplitude):
        keys = key_handler.all['diff']

        target = ift.DomainTuple.make(target)
        if len(target) > 1:
            raise ValueError
        h_space = target[0].get_default_codomain()
        ht = ift.HarmonicTransformOperator(h_space, target[0])
        p_space = amplitude.target[0]
        power_distributor = ift.PowerDistributor(h_space, p_space)

        vol = h_space.scalar_dvol**-0.5
        self._amplitude = vol*amplitude
        self._A = power_distributor @ self._amplitude
        self._xi = ift.ducktape(h_space, None, keys['x'])
        corr_lin_fld = ht(self._A*self._xi)

        self._op = corr_lin_fld.clip(-40, 40).exp()

    @property
    def pspec(self):
        return self._amplitude**2

    def pre_image(self, field):
        if not isinstance(field, ift.Field):
            raise TypeError
        sp = self.target[0]
        a = self._amplitude
        keys = key_handler.all['diff']

        hsp = sp.get_default_codomain()
        ht = ift.HartleyOperator(hsp, target=sp)
        sep_logd_h = ht.inverse(field.log())
        A = ift.PowerDistributor(hsp, power_space=a.target[0]) @ a

        p = ift.full(self.domain, 0.).to_dict()
        p[keys['x']] = sep_logd_h/A.force(ift.MultiField.from_dict(p))
        return ift.MultiField.from_dict(p)
