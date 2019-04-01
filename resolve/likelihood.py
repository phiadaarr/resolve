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

from functools import reduce
from operator import mul

import nifty5 as ift

from .calibration_distributor import CalibrationDistributor


def sqrt_n_operator(var, key):
    dom = var.domain
    scale_q = ift.DiagonalOperator(1/(0.5*var))
    N_amp = scale_q @ ift.InverseGammaOperator(dom, alpha=0.5, q=1) @ scale_q
    return ift.Realizer(dom) @ N_amp.ducktape(key)**-0.5


def make_calibration(dh, cal_ops):
    dom = list(cal_ops.values())[0].target

    dtr_a1 = CalibrationDistributor(dom, dh.ant1, dh.time)
    dtr_a2 = CalibrationDistributor(dom, dh.ant2, dh.time)
    dtr_ampl = dtr_a1 + dtr_a2
    dtr_ph = 1j*(dtr_a1 - dtr_a2)

    ops_pol0, ops_pol1 = [], []
    for mode, dtr in [('ampl', dtr_ampl), ('ph', dtr_ph)]:
        try:
            ops_pol0.append((dtr @ cal_ops['{}0'.format(mode)]).exp())
            ops_pol1.append((dtr @ cal_ops['{}1'.format(mode)]).exp())
        except KeyError:
            pass

    nrows, nchans = dh.vis.shape[0:2]
    dom = ift.DomainTuple.make((ift.UnstructuredDomain(nrows),
                                ift.UnstructuredDomain(nchans)))
    broadcast_channels = ift.ContractionOperator(dom, 1).adjoint
    tgt = ift.DomainTuple.make((dom[0], dom[1], ift.UnstructuredDomain(2)))
    insert_pol0 = ift.DomainTupleFieldInserter(tgt, 2, (0,))
    insert_pol1 = ift.DomainTupleFieldInserter(tgt, 2, (1,))

    pol0 = insert_pol0 @ broadcast_channels @ reduce(mul, ops_pol0)
    pol1 = insert_pol1 @ broadcast_channels @ reduce(mul, ops_pol1)
    return pol0 + pol1


def make_signal_response(dh, R, sky, cal_ops):
    if isinstance(sky, ift.Field) and len(cal_ops) == 0:
        raise NotImplementedError
    rsky = R(sky)
    if len(cal_ops) > 0:
        cop = make_calibration(dh, cal_ops)
        if isinstance(sky, ift.Field):
            return ift.makeOp(rsky) @ cop
        elif isinstance(sky, ift.Operator):
            return cop*rsky
        else:
            raise RuntimeError
    return rsky


def make_likelihood(dh, R, sky, cal_ops):
    resi = ift.Adder(-dh.vis) @ make_signal_response(dh, R, sky, cal_ops)
    var = ift.makeOp(dh.var)
    # if noise_inference:
    #     resi = sqrt_n_operator(dh.var, key_handler.all['eta'][key])*resi
    #     var = ift.ScalingOperator(1., dh.var.domain)
    return ift.GaussianEnergy(domain=resi.target, covariance=var) @ resi


