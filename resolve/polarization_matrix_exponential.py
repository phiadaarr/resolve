# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift


def polarization_matrix_exponential(domain, with_v, jax=False):
    dom = ift.makeDomain(domain)
    keys = ["i", "q", "u"]
    if with_v:
        keys += ["v"]
    domain = ift.makeDomain({kk: dom for kk in keys})
    target = ift.makeDomain({kk.upper(): dom for kk in keys})
    if jax:
        return _jax_pol(domain, target)
    return PolarizationMatrixExponential(domain, target)


class PolarizationMatrixExponential(ift.Operator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)

    def apply(self, x):
        self._check_input(x)
        with_v = "v" in self.domain.keys()
        duckI = ift.ducktape(None, self._domain["i"], "I")
        duckQ = ift.ducktape(None, self._domain["q"], "Q")
        duckU = ift.ducktape(None, self._domain["u"], "U")
        tmpi = x["i"].exp()
        if with_v:
            duckV = ift.ducktape(None, self._domain["u"], "V")
            log_p = (x["q"] ** 2 + x["u"] ** 2 + x["v"] ** 2).sqrt()
        else:
            log_p = (x["q"] ** 2 + x["u"] ** 2).sqrt()
        I = duckI(tmpi * log_p.cosh())
        tmp = tmpi * log_p.sinh() * log_p.reciprocal()
        U = duckU(tmp * x["u"])
        Q = duckQ(tmp * x["q"])
        if with_v:
            V = duckV(tmp * x["v"])
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


def _jax_pol(domain, target):
    from jax.numpy import sqrt, exp, cosh, sinh
    with_v = "v" in domain.keys()

    def func(x):
        res = {}
        sq = x["q"] ** 2 + x["u"] ** 2
        if with_v:
            sq = sq + x["v"] ** 2
        log_p = sqrt(sq)
        tmpi = exp(x["i"])
        res["I"] = tmpi * cosh(log_p)
        tmp = tmpi * sinh(log_p) / log_p
        res["U"] = tmp * x["u"]
        res["Q"] = tmp * x["q"]
        if with_v:
            res["V"] = tmp * x["v"]
        return res

    return ift.JaxOperator(domain, target, func)
