import resolve as rve
import numpy as np
import resolve_support
import nifty8 as ift


def my_operator(ant1, ant2, time, tdom, pdom, fdom, key_phase, key_logampl, antenna_dct, nthreads):
    target = pdom, ift.UnstructuredDomain(len(ant1)), fdom
    ant1 = rve.replace_array_with_dict(ant1, antenna_dct)
    ant2 = rve.replace_array_with_dict(ant2, antenna_dct)
    nants = len(set(ant1).union(set(ant2)))
    dom = pdom, ift.UnstructuredDomain(nants), tdom, fdom
    dom = {kk: dom for kk in [key_phase, key_logampl]}
    return rve.Pybind11Operator(
        dom,
        target,
        resolve_support.CalibrationDistributor(
            ant1.astype(np.int32),
            ant2.astype(np.int32),
            time,
            key_logampl,
            key_phase,
            fdom.size,
            tdom.size,
            tdom.distances[0],
            nthreads,
        ),
    )


def test_calibration_distributor():
    nt = 10

    rng = np.random.default_rng(42)

    fdom = ift.UnstructuredDomain(1)
    tmax = 2
    ant1 = np.array([0, 0, 1]).astype(np.int32)
    ant2 = np.array([1, 3, 3]).astype(np.int32)
    time = np.array([0, 1.0, 2.5])
    nrow = len(ant1)

    polarization = rve.Polarization([5])
    npol = 1
    freq = np.array([1.2e9])
    nfreq = len(freq)

    uvw = rng.random((nrow, 3)) - 0.5
    weight = rng.uniform(0.9, 1.1, (npol, nrow, nfreq)).astype(np.float32)
    vis = (
        rng.random((npol, nrow, nfreq)) - 0.5 + 1j * (rng.random((npol, nrow, nfreq)) - 0.5)
    ).astype(np.complex64)
    obs = rve.Observation(
        rve.AntennaPositions(uvw, ant1, ant2, time), vis, weight, polarization, freq
    )

    pdom = obs.vis.domain[0]
    fdom = obs.vis.domain[2]

    tdom = ift.RGSpace(nt, tmax / nt * 2)

    key_phase, key_logampl = "p", "a"

    antenna_dct = {aa: ii for ii, aa in enumerate(rve.unique_antennas(obs))}

    nthreads = 1
    args = ant1, ant2, time, tdom, pdom, fdom, key_phase, key_logampl, antenna_dct, nthreads

    idop = ift.Operator.identity_operator(
        (pdom, ift.UnstructuredDomain(len(antenna_dct)), tdom, fdom)
    )
    # reference operator
    op0 = rve.calibration_distribution(
        obs, idop.ducktape(key_phase), idop.ducktape(key_logampl), antenna_dct, numpy=True
    )
    op1 = rve.calibration_distribution(
        obs, idop.ducktape(key_phase), idop.ducktape(key_logampl), antenna_dct
    )
    op = my_operator(*args)

    rve.operator_equality(op0, op)
