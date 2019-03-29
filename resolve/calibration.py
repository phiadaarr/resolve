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

from .amplitude_operators import LAmplitude
from .extended_operator import AXiOperator


class Calibration(AXiOperator):
    def __init__(self,
                 *,
                 t_pix,
                 t_max,
                 antennas,
                 xi_key,
                 zero_padding_factor,
                 amplitude,
                 clip=[]):
        tdst = t_max/(t_pix - 1)
        print('Length of time steps for calibration solutions: {0:.1f} min'.
              format(tdst/60))
        tspace = ift.RGSpace(t_pix, distances=tdst)

        sp_ant = ift.UnstructuredDomain(len(antennas))
        dom = ift.DomainTuple.make((sp_ant, tspace))
        zp = ift.FieldZeroPadder(
            dom, (zero_padding_factor*tspace.shape[0],), space=1)
        h_space = ift.DomainTuple.make((zp.target[0],
                                        zp.target[1].get_default_codomain()))
        vol = h_space[-1].scalar_dvol**-0.5
        ht = ift.HarmonicTransformOperator(
            h_space, space=1, target=zp.target[1])
        pd = ift.PowerDistributor(h_space[-1])
        bc = ift.ContractionOperator(h_space, 0).adjoint

        if not isinstance(amplitude, ift.Operator):
            amplitude['target'] = pd.domain
            amplitude = LAmplitude(**amplitude)

        self._amplitude = amplitude
        self._A = bc @ pd @ (vol*self._amplitude)
        self._xi = ift.ducktape(h_space, None, xi_key)
        hop = self._A*self._xi

        if len(clip) not in [0, 2]:
            raise ValueError

        self._op = zp.adjoint @ ht @ hop
        if len(clip) == 2:
            self._op = self._op.clip(*clip)
        self.nozeropad = ht @ hop

    @property
    def pspec(self):
        return self._amplitude**2

    @property
    def amplitude(self):
        return self._amplitude
