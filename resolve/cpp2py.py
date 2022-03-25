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
# Copyright(C) 2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras, Jakob Roth

import nifty8 as ift
import numpy as np
from functools import partial
from .logger import logger


class Pybind11Operator(ift.Operator):
    def __init__(self, dom, tgt, op, nifty_equivalent=None):
        self._domain = ift.makeDomain(dom)
        self._target = ift.makeDomain(tgt)
        self._op = op
        self._nifty_equivalent = nifty_equivalent

    def apply(self, x):
        self._check_input(x)
        if ift.is_linearization(x):
            lin = self._op.apply_with_jac(x.val.val)
            jac = Pybind11LinearOperator(
                self._domain, self._target, lin.jac_times, lin.jac_adjoint_times
            )
            pos = ift.makeField(self._target, lin.position())
            return x.new(pos, jac)
        return ift.makeField(self.target, self._op.apply(x.val))

    @property
    def nifty_equivalent(self):
        return self._nifty_equivalent


class Pybind11LikelihoodEnergyOperator(ift.LikelihoodEnergyOperator):
    def __init__(
        self,
        dom,
        op,
        draw_sample=None,
        get_transformation=None,
        data_residual=None,
        sqrt_data_metric=None,
        nifty_equivalent=None,
    ):
        """

        Parameters
        ----------
        dom :

        op :

        draw_sample :

        get_transformation :

        data_residual :

        sqrt_data_metric :

        nifty_equivalent : nifty8.LikelihoodEnergyOperator, optional
            NIFTy operator that represents the same functionality as the
            operator that is implemented in C++. If `data_residual`,
            `sqrt_data_metric`, `draw_sample`, `get_transformation` are not
            given explicitly they are replaced by `nifty_equivalent`. default
            None.
        """
        self._domain = ift.makeDomain(dom)
        self._op = op
        if data_residual is None and nifty_equivalent is not None:
            logger.info("Use nifty equivalent for data_residual")
            data_residual = nifty_equivalent._res
        if sqrt_data_metric is None and nifty_equivalent is not None:
            logger.info("Use nifty equivalent for sqrt_data_metric")
            sqrt_data_metric = nifty_equivalent._sqrt_data_metric_at
        if draw_sample is None:
            logger.info("Use nifty equivalent for draw_sample")
            draw_sample = lambda loc, from_inverse: nifty_equivalent(
                ift.Linearization.make_var(ift.makeField(self._domain, loc), True)
            ).metric.draw_sample(from_inverse)
        if get_transformation is None:
            logger.info("Use nifty equivalent for get_transformation")
            get_transformation = nifty_equivalent.get_transformation
        self._draw_sample = draw_sample
        self.get_transformation = get_transformation
        self._nifty_equivalent = nifty_equivalent
        super(Pybind11LikelihoodEnergyOperator, self).__init__(
            data_residual, sqrt_data_metric
        )

    def apply(self, x):
        self._check_input(x)
        if not ift.is_linearization(x):
            return ift.makeField(self.target, self._op.apply(x.val))
        lin = self._op.apply_with_jac(x.val.val)
        jac = Pybind11LinearOperator(
            self._domain, self._target, lin.jac_times, lin.jac_adjoint_times
        )
        pos = ift.makeField(self._target, lin.position())
        res = x.new(pos, jac)
        if x.want_metric:
            metric = Pybind11SelfAdjointOperator(
                self.domain, lin.apply_metric, partial(self._draw_sample, x.val.val)
            )
            res = res.add_metric(metric)
        return res

    def get_metric_at(self, x):
        raise NotImplementedError

    @property
    def nifty_equivalent(self):
        return self._nifty_equivalent


class Pybind11LinearOperator(ift.LinearOperator):
    def __init__(self, domain, target, times, adj_times, draw_sample=None):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._times = times
        self._adj_times = adj_times
        self.draw_sample = draw_sample

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = (self._times if mode == self.TIMES else self._adj_times)(x.val)
        return ift.makeField(self._tgt(mode), res)


def Pybind11SelfAdjointOperator(domain, times, draw_sample=None):
    return Pybind11LinearOperator(domain, domain, times, times, draw_sample)
