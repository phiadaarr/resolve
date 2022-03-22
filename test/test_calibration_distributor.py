import resolve as rve
import numpy as np
import jax.numpy as jnp
import resolvelib
import nifty8 as ift


def my_operator(ant1, ant2, time, tdom, pdom, fdom, key_phase, key_logampl, antenna_dct):
    target = pdom, ift.UnstructuredDomain(len(ant1)), fdom
    ant1 = rve.replace_array_with_dict(ant1, antenna_dct)
    ant2 = rve.replace_array_with_dict(ant2, antenna_dct)
    nants = len(set(ant1).union(set(ant2)))
    dom = pdom, ift.UnstructuredDomain(nants), tdom, fdom
    dom = {kk: dom for kk in [key_phase, key_logampl]}
    return rve.Pybind11Operator(
            dom,
        target,
        resolvelib.CalibrationDistributor(
            ant1.astype(np.int32), ant2.astype(np.int32), time, key_logampl, key_phase, fdom.size, tdom.size, tdom.distances[0]
        ),
    )

def test_calibration_distributor():
    obs = next(rve.ms2observations_all("~/data/CYG-ALL-2052-2MHZ.ms", "DATA"))
    #obs = obs[:50]
    tmin, tmax = rve.tmin_tmax(obs)
    obs = obs.move_time(-tmin)
    tmin, tmax = rve.tmin_tmax(obs)

    ant1 = obs.ant1
    ant2 = obs.ant2
    time = obs.time

    pdom = obs.vis.domain[0]
    fdom = obs.vis.domain[2]

    nt = 10
    tdom = ift.RGSpace(nt, tmax/nt*2)

    key_phase, key_logampl = "p", "a"

    antenna_dct = {aa: ii for ii, aa in enumerate(rve.unique_antennas(obs))}

    args = ant1, ant2, time, tdom, pdom, fdom, key_phase, key_logampl, antenna_dct

    idop = ift.Operator.identity_operator((pdom, ift.UnstructuredDomain(len(antenna_dct)), tdom, fdom))
    op0 = rve.calibration_distribution(obs, idop.ducktape(key_phase), idop.ducktape(key_logampl), antenna_dct)  # reference operator

    op = my_operator(*args)
    rve.operator_equality(op0, op, rtol=1e-5, atol=1e-5)
