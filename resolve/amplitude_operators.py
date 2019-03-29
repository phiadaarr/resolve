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

import numpy as np

import nifty5 as ift

from .zero_mode_operator import ZeroModeOperator


def LAmplitude(*, target, n_pix, sm, sv, im, iv, alpha, q, linear_key,
               zeromode_key):
    et = ift.ExpTransform(target, n_pix)
    dom = et.domain[0]
    sl = ift.SlopeOperator(dom)
    mean = np.array([sm, im + sm*dom.t_0[0]])
    sig = np.array([sv, iv])
    mean = ift.Field.from_global_data(sl.domain, mean)
    sig = ift.Field.from_global_data(sl.domain, sig)
    linear = sl @ ift.Adder(mean) @ ift.makeOp(sig).ducktape(linear_key)
    a = et @ (0.5*linear).exp()
    return _zmmaskit(a, alpha, q, zeromode_key)


def SLAmplitude(*, slamplcfg, alpha, q, zeromode_key):
    a = ift.SLAmplitude(**slamplcfg)
    return _zmmaskit(a, alpha, q, zeromode_key)


def _zmmaskit(amplitude, alpha, q, key):
    dom = amplitude.target
    zm = ZeroModeOperator(dom, alpha, q, key)
    mask = np.ones(dom.shape)
    mask[0] = 0
    mask = ift.makeOp(ift.from_global_data(dom, mask))
    return mask @ amplitude + zm
