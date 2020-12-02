# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

from os.path import join

import numpy as np
from ducc0.fft import good_size

import nifty7 as ift
import resolve as rve


def main():
    rve.set_nthreads(8)
    rve.set_wgridding(True)
    obs = rve.ms2observations(
        "data/AM754_A030124_flagged.ms", "DATA", True, 0, "stokesi"
    )
    t0, _ = rve.tmin_tmax(*obs)
    obs = [oo.move_time(-t0) for oo in obs]
    ocalib1, ocalib2, oscience = obs
    rve.set_epsilon(min(1 / 10 / oo.max_snr() for oo in obs))

    uantennas = rve.unique_antennas(*obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uantennas)}
    npol, nfreq = obs[0].npol, obs[0].nfreq

    total_N = npol * len(uantennas) * nfreq
    tmin, tmax = rve.tmin_tmax(*obs)
    solution_interval = 20  # s
    time_domain = ift.RGSpace(
        good_size(int(2 * (tmax - tmin) / solution_interval)), solution_interval
    )
    print(f"Npix in time domain {time_domain.shape[0]}")
    reshaper = rve.Reshaper(
        [ift.UnstructuredDomain(total_N), time_domain],
        [
            ift.UnstructuredDomain(npol),
            ift.UnstructuredDomain(len(uantennas)),
            time_domain,
            ift.UnstructuredDomain(nfreq),
        ],
    )
    dct = {"offset_mean": 0, "offset_std": (1, 0.5), "prefix": "calibration_phases"}
    dct1 = {
        "fluctuations": (2.0, 1.0),
        "loglogavgslope": (-4.0, 1),
        "flexibility": (5, 2.0),
        "asperity": None,
    }
    cfmph = ift.CorrelatedFieldMaker.make(**dct, total_N=total_N)
    cfmph.add_fluctuations(time_domain, **dct1)
    phase = reshaper @ cfmph.finalize(0)
    dct = {
        "offset_mean": 0,
        "offset_std": (1e-3, 1e-6),
        "prefix": "calibration_logamplitudes",
    }
    dct1 = {
        "fluctuations": (2.0, 1.0),
        "loglogavgslope": (-4.0, 1),
        "flexibility": (5, 2.0),
        "asperity": None,
    }
    cfmampl = ift.CorrelatedFieldMaker.make(**dct, total_N=total_N)
    cfmampl.add_fluctuations(time_domain, **dct1)
    logampl = reshaper @ cfmampl.finalize(0)

    fov = np.array([2, 2]) * rve.DEG2RAD
    npix = np.array([1024, 1024])
    skydom = ift.RGSpace(npix, fov / npix)
    dct = {
        "target": skydom,
        "offset_mean": 19,
        "offset_std": (1, 0.1),
        "prefix": "logdiffuse",
        "fluctuations": (5.0, 1.0),
        "loglogavgslope": (-2.0, 1),
        "flexibility": (1.0, 0.5),
        "asperity": None,
    }
    logsky = ift.SimpleCorrelatedField(**dct)
    sky = logsky.exp()

    abc_calib2 = rve.calibration_distribution(
        ocalib2, phase, logampl, antenna_dct, None
    )
    lh0 = rve.CalibrationLikelihood(
        ocalib2, abc_calib2, ift.full(ocalib2.vis.domain, 1 + 0.0j)
    )
    abc_science = rve.calibration_distribution(
        oscience, phase, logampl, antenna_dct, None
    )
    lh1 = rve.ImagingCalibrationLikelihood(oscience, sky, abc_science)

    plotter = rve.Plotter("png", "plots")
    plotter.add_calibration_solution("calibration_logamplitude", logampl)
    plotter.add_calibration_solution("calibration_phase", phase)
    plotter.add_histogram("normalized_residuals_calib2", lh0.normalized_residual)
    plotter.add_histogram("normalized_residuals_science", lh1.normalized_residual)
    ham = ift.StandardHamiltonian(lh0)
    state = rve.MinimizationState(0.1 * ift.from_random(ham.domain), [])
    minimizer = ift.NewtonCG(
        ift.GradientNormController(name="newton", iteration_limit=10)
    )
    n0, n1 = 3, 3
    for ii in range(n0):
        state = rve.simple_minimize(ham, state.mean, 0, minimizer)
        plotter.plot(f"{ii}", state)

    # FIXME Total flux is unconstrained
    ham = ift.StandardHamiltonian(lh1)
    state = rve.MinimizationState(
        ift.MultiField.union([0.1 * ift.from_random(ham.domain), state.mean]), []
    )
    plotter.add("logsky", sky.log10())
    plotter.add("logsky_pspec", logsky.power_spectrum)

    for ii in range(n0, n0 + n1):
        state = rve.simple_minimize(ham, state.mean, 0, minimizer)
        plotter.plot(f"{ii}", state)
        state.save(join(plotter.directory, f"{ii}.state"))


if __name__ == "__main__":
    main()
