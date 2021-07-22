# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift


class polarization_matrix_exponential(ift.Operator):
    def __init__(self, domain, with_v):
        self._domain = ift.makeDomain(domain)
        keys = ["i", "q", "u"]
        if with_v:
            keys += ["v"]
        assert set(self._domain.keys()) == set(keys)
        assert self._domain["i"] == self._domain["q"] == self._domain["u"]
        if with_v:
            assert self._domain["i"] == self._domain["v"]
        self._target = ift.makeDomain(
            {kk.upper(): self._domain["i"] for kk in self._domain.keys()}
        )
        self._with_v = with_v

    def apply(self, x):
        self._check_input(x)
        duckI = ift.ducktape(None, self._domain["i"], "I")
        duckQ = ift.ducktape(None, self._domain["q"], "Q")
        duckU = ift.ducktape(None, self._domain["u"], "U")
        tmpi = x["i"].exp()
        if self._with_v:
            duckV = ift.ducktape(None, self._domain["u"], "V")
            log_p = (x["q"] ** 2 + x["u"] ** 2 + x["v"] ** 2).sqrt()
        else:
            log_p = (x["q"] ** 2 + x["u"] ** 2).sqrt()
        I = duckI(tmpi * log_p.cosh())
        tmp = tmpi * log_p.sinh() * log_p.reciprocal()
        U = duckU(tmp * x["u"])
        Q = duckQ(tmp * x["q"])
        if self._with_v:
            V = duckV(tmp * x["v"])
        if ift.is_linearization(x):
            val = I.val.unite(U.val.unite(Q.val))
            jac = I.jac + U.jac + Q.jac
            if self._with_v:
                val = val.unite(V.val)
                jac = jac + V.jac
            return x.new(val, jac)
        if self._with_v:
            return I.unite(U.unite(Q.unite(V)))
        return I.unite(U.unite(Q))
