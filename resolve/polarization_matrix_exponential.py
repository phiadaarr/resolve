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
# Copyright(C) 2019-2021 Max-Planck-Society
# Copyright(C) 2019-2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

from warnings import warn

import nifty8 as ift

import resolve_support

from .cpp2py import Pybind11Operator
from .polarization_space import PolarizationSpace
from .simple_operators import MultiFieldStacker


def polarization_matrix_exponential_mf2f(domain, nthreads=1):
    """

    Note
    ----
    In contrast to polarization_matrix_exponential this takes a MultiField as
    an input and returns a Field.
    """
    domain = ift.MultiDomain.make(domain)
    if list(domain.keys()) == ["I"]:
        return ift.ducktape(domain, None, "I").adjoint.exp()
    pdom = PolarizationSpace(["I", "Q", "U", "V"])
    assert pdom.labels_eq(domain.keys())
    restdom = domain.values()[0]
    assert all(dd == restdom for dd in domain.values())
    target = (pdom,) + tuple(restdom)
    if len(restdom.shape) == 1:
        f = resolve_support.PolarizationMatrixExponential1
    elif len(restdom.shape) == 2:
        f = resolve_support.PolarizationMatrixExponential2
    elif len(restdom.shape) == 3:
        f = resolve_support.PolarizationMatrixExponential3
    elif len(restdom.shape) == 4:
        f = resolve_support.PolarizationMatrixExponential4
    else:
        raise NotImplementedError("Not compiled for this shape")
    return Pybind11Operator(domain, target, f(nthreads))


def polarization_matrix_exponential(domain):
    """

    Deprecated.

    Parameters
    ----------
    domain : DomainTuple
        DomainTuple of which the first entry is a PolarizationSpace.
    """
    warn("polarization_matrix_exponential is deprecated. "
         "Use polarization_matrix_exponential_mf2f instead.", DeprecationWarning)
    dom = ift.DomainTuple.make(domain)
    pdom = dom[0]
    assert isinstance(pdom, PolarizationSpace)

    if pdom.labels_eq("I"):
        return ift.ScalingOperator(domain, 1.).exp()

    mfs = MultiFieldStacker(domain, 0, domain[0].labels)
    op = PolarizationMatrixExponential(mfs.domain)
    return mfs @ op @ mfs.inverse


class PolarizationMatrixExponential(ift.Operator):
    def __init__(self, domain):
        self._domain = self._target = ift.makeDomain(domain)
        assert set(self._domain.keys()) in [set(["I", "Q", "U"]), set(["I", "Q", "U", "V"])]

    def apply(self, x):
        self._check_input(x)
        with_v = "V" in self.domain.keys()
        tmpi = x["I"].exp()
        if with_v:
            log_p = (x["Q"] ** 2 + x["U"] ** 2 + x["V"] ** 2).sqrt()
        else:
            log_p = (x["Q"] ** 2 + x["U"] ** 2).sqrt()
        I = tmpi * log_p.cosh()
        tmp = tmpi * log_p.sinh() * log_p.reciprocal()
        U = tmp * x["U"]
        Q = tmp * x["Q"]
        if with_v:
            V = tmp * x["V"]
        I = ift.ducktape(None, self._domain["I"], "I")(I)
        Q = ift.ducktape(None, self._domain["Q"], "Q")(Q)
        U = ift.ducktape(None, self._domain["U"], "U")(U)
        if with_v:
            V = ift.ducktape(None, self._domain["V"], "V")(V)
        if ift.is_linearization(x):
            val = I.val.unite(U.val.unite(Q.val))
            jac = I.jac + U.jac + Q.jac
            if with_v:
                val = val.unite(V.val)
                jac = jac + V.jac
            return x.new(val, jac)
        if with_v:
            return I.unite(U.unite(Q.unite(V)))
        return I.unite(U.unite(Q))
