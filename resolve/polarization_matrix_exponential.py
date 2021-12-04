# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift

from .polarization_space import PolarizationSpace
from .simple_operators import MultiFieldStacker


def polarization_matrix_exponential(domain, jax=False):
    dom = ift.makeDomain(domain)
    pdom = dom[0]
    assert isinstance(pdom, PolarizationSpace)
    if pdom.labels_eq("I"):
        return ift.ScalingOperator(domain, 1.)

    mfs = MultiFieldStacker(domain, 0, domain[0].labels)
    if jax:
        op = _jax_pol(mfs.domain)
    else:
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


def _jax_pol(domain):
    from jax.numpy import cosh, exp, sinh, sqrt
    with_v = "V" in domain.keys()

    def func(x):
        res = {}
        sq = x["Q"] ** 2 + x["U"] ** 2
        if with_v:
            sq = sq + x["V"] ** 2
        log_p = sqrt(sq)
        tmpi = exp(x["I"])
        res["I"] = tmpi * cosh(log_p)
        tmp = tmpi * sinh(log_p) / log_p
        res["U"] = tmp * x["U"]
        res["Q"] = tmp * x["Q"]
        if with_v:
            res["V"] = tmp * x["V"]
        return res

    return ift.JaxOperator(domain, domain, func)
