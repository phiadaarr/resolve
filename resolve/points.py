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

from .configuration_parser import ConfigurationParser
from .extended_operator import ExtendedOperator
from .key_handler import key_handler


class Points(ExtendedOperator):
    def __init__(self, cfg):
        if not isinstance(cfg, ConfigurationParser):
            raise TypeError
        c = cfg.points
        if not set(['alpha', 'q']) <= set(c):
            raise ValueError

        tgt = cfg.sky_space
        key = key_handler.all['points']
        alpha = float(c.getfloat('alpha'))
        q = float(c.getfloat('q'))
        self._op = ift.InverseGammaOperator(
            tgt, alpha=alpha, q=q).ducktape(key)
        self._alpha = alpha
        self._q = q

    def pre_image(self, field):
        if not isinstance(field, ift.Field):
            raise TypeError
        return ift.InverseGammaModel.inverseIG(
            field, alpha=self._alpha, q=self._q)
