import sys

import nifty8 as ift
import numpy as np
import resolve as rve
import resolve_support


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


def main():
    if True:
        obs = next(rve.ms2observations_all("/data/CYG-ALL-2052-2MHZ.ms", "DATA"))
    else:
        obs = next(rve.ms2observations_all("/data/meerkat-calibration/ms1_primary_subset.ms", "DATA"))
        if True:
            obs = obs.remove_autocorrelations().to_double_precision()
        else:
            obs = obs.remove_autocorrelations().restrict_to_polarization("XX").to_double_precision()

    quick = False
    if len(sys.argv) == 2 and sys.argv[1] == "quick":
        quick = True
        obs = obs[:50]
    tmin, tmax = rve.tmin_tmax(obs)
    obs = obs.move_time(-tmin)
    tmin, tmax = rve.tmin_tmax(obs)

    ant1 = obs.ant1
    ant2 = obs.ant2
    time = obs.time

    pdom = obs.vis.domain[0]
    fdom = obs.vis.domain[2]

    nt = 2*559
    tdom = ift.RGSpace(nt, tmax / nt * 2)

    key_phase, key_logampl = "p", "a"

    antenna_dct = {aa: ii for ii, aa in enumerate(rve.unique_antennas(obs))}

    args = ant1, ant2, time, tdom, pdom, fdom, key_phase, key_logampl, antenna_dct

    idop = ift.Operator.identity_operator(
        (pdom, ift.UnstructuredDomain(len(antenna_dct)), tdom, fdom)
    )
    # reference operator
    op0 = rve.calibration_distribution(
        obs, idop.ducktape(key_phase), idop.ducktape(key_logampl), antenna_dct
    )


    print("Old implementation")
    ift.exec_time(op0)
    for nthreads in range(1, 5):
        print(f"New implementation (nthreads={nthreads})")
        op = my_operator(*args, nthreads=nthreads)
        ift.exec_time(op, ntries=10)

        if quick:
            pos = ift.from_random(op.domain)
            ift.extra.check_operator(op, pos, ntries=3)
            ift.extra.check_operator(op0, pos, ntries=3)

            rve.operator_equality(op0, op, rtol=1e-5, atol=1e-5, ntries=1)


if __name__ == "__main__":
    main()
