import resolve as rve
import numpy as np
import jax.numpy as jnp
import resolvelib
import nifty8 as ift


def reference_operator(ant1, ant2, time, tdom, pdom, fdom, key_phase, key_logampl):
    nants = len(set(ant1).union(set(ant2)))
    target = pdom, ift.UnstructuredDomain(len(ant1)), fdom
    dom = ift.UnstructuredDomain(nants), tdom, fdom
    dom = {kk: dom for kk in [key_phase, key_logampl]}

    def func(x):
        y0 = x[key_logampl][ant1, time] + x[key_logampl][ant2, time]
        y1 = x[key_phase][ant1, time] - x[key_phase][ant2, time]
        return jnp.exp(y0 + 1j * y1)[None]

    inp = ift.Operator.identity_operator(dom)
    tmp = inp.ducktape_left(key_logampl) + inp.ducktape_left(key_phase)
    return ift.JaxOperator(dom, target, func)


def my_operator(ant1, ant2, time, tdom, pdom, fdom, key_phase, key_logampl):
    nants = len(set(ant1).union(set(ant2)))
    target = pdom, ift.UnstructuredDomain(len(ant1)), fdom
    dom = ift.UnstructuredDomain(nants), tdom, fdom
    dom = {kk: dom for kk in [key_phase, key_logampl]}
    return rve.Pybind11Operator(
            dom,
        target,
        resolvelib.CalibrationDistributor(
            ant1, ant2, time, key_logampl, key_phase, fdom.size
        ),
    )

def test_calibration_distributor():
    nfreq = 2

    ant1 = np.array([0, 0, 0]).astype(np.int32)
    ant2 = np.array([1, 2, 2]).astype(np.int32)
    time = np.array([0, 0, 1]).astype(np.int32)

    pdom = ift.UnstructuredDomain(1)
    fdom = ift.UnstructuredDomain(nfreq)
    tdom = ift.RGSpace(10, 0.2)

    key_phase, key_logampl = "p", "a"

    args = ant1, ant2, time, tdom, pdom, fdom, key_phase, key_logampl

    op0 = reference_operator(*args)
    op = my_operator(*args)
    rve.operator_equality(op0, op)
