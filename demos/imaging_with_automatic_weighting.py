# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import configparser
import os
import sys
from distutils.util import strtobool

import nifty8 as ift
import numpy as np
import resolve as rve
from matplotlib.colors import LogNorm
from resolve.mpi import master


def main(cfg_file_name):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file_name)
    if master:
        print(f"Load {cfg_file_name}")

    rve.set_epsilon(cfg["response"].getfloat("epsilon"))
    rve.set_wgridding(strtobool(cfg["response"]["wgridding"]))
    rve.set_double_precision(True)
    rve.set_nthreads(cfg["technical"].getint("nthreads"))

    enable_weighting = cfg["weighting"].getboolean("enable")

    # Data
    obs_file_name = cfg["data"]["path"]
    if os.path.splitext(obs_file_name)[1] == ".npz":
        obs = rve.Observation.load(obs_file_name).restrict_to_stokesi().average_stokesi()
    elif os.path.splitext(obs_file_name)[1] == ".ms":
        obs = rve.ms2observations(obs_file_name, "DATA", False, 0, "stokesiavg")
        assert len(obs) == 1
        obs = obs[0]
    else:
        raise RuntimeError(
            f"Do not understand file name ending of {obs_file_name}. Supported: ms, npz.")
    # /Data

    # Sky model
    sky, operators = rve.single_frequency_sky(cfg["sky"])
    raw_sky = operators["sky"]
    enable_points = operators["points"] is not None
    operators["logsky"] = raw_sky.log()

    p = ift.Plot()
    for _ in range(9):
        p.add(raw_sky(ift.from_random(raw_sky.domain)), norm=LogNorm())
    if master:
        p.output(name="sky_prior_samples.png")
    # /Sky model

    # Bayesian weighting
    if enable_weighting:
        assert obs.nfreq == obs.npol == 1
        subcfg = cfg["weighting"]
        if subcfg["model"] == "cfm":
            import ducc0

            npix = subcfg.getint("npix")
            fac = subcfg.getfloat("zeropadding factor")
            npix_padded = ducc0.fft.good_size(int(np.round(npix*fac)))

            xs = np.log(obs.effective_uvwlen().val)
            xs -= np.min(xs)
            if not xs.shape[0] == xs.shape[2] == 1:
                raise RuntimeError
            dom = ift.RGSpace(npix_padded, fac * np.max(xs) / npix)
            logwgt, cfm = rve.cfm_from_cfg(subcfg, dom, "invcov")
            li = ift.LinearInterpolator(dom, xs[0].T)
            conv = rve.DomainChangerAndReshaper(li.target, obs.weight.domain)
            weightop = ift.makeOp(obs.weight) @ (conv @ li @ logwgt.exp()) ** (-2)
            operators["logweights"] = logwgt
            operators["weights"] = logwgt.exp()
            operators["logweights power spectrum"] = cfm.power_spectrum
        else:
            raise NotImplementedError
    else:
        weightop = None
    # /Bayesian weighting

    full_lh = rve.ImagingLikelihood(obs, sky, inverse_covariance_operator=weightop)
    position = 0.1 * ift.from_random(full_lh.domain)

    common = {"plottable_operators": operators, "overwrite": True, "return_final_position": True}

    def get_mini(iglobal):
        if iglobal == 0:
            return ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=4))
        if iglobal == 1:
            return ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=20))
        return ift.VL_BFGS(ift.GradientNormController(name="bfgs", iteration_limit=20))

    def get_sampling(iglobal):
        if iglobal in [0, 1]:
            return None
        return ift.AbsDeltaEnergyController(deltaE=0.5, convergence_level=3, iteration_limit=500,
                                            name="Sampling")

    def get_cst(iglobal):
        if iglobal in [0, 1]:
            return []
        return sky.domain.keys()

    def get_lh(iglobal):
        if iglobal == 0:
            return rve.ImagingLikelihood(obs, operators["points"])
        if iglobal == 1:
            return rve.ImagingLikelihood(obs, sky)
        return full_lh

    def get_n_samples(iglobal):
        if iglobal in [0, 1]:
            return 0
        return 3

    def get_comm(iglobal):
        return rve.mpi.comm

    def callback(sl, iglobal):
        lh = get_lh(iglobal)
        s = ift.extra.minisanity(lh.data, lh.metric_at_pos, lh.model_data, sl)
        if rve.mpi.master:
            print(s)

    if True:
        _, position = ift.optimize_kl(get_lh, 7, get_n_samples, get_mini, get_sampling, None,
                                      constants=get_cst, point_estimates=get_cst,
                                      initial_position=position, comm=get_comm, callback=callback,
                                      output_directory=cfg["output"]["directory"] + "_initial",
                                      **common)
    else:
        position = ift.ResidualSampleList.load_mean("Cygnus_A_2052MHz_initial/pickle/last")

    # Reset diffuse component
    position = ift.MultiField.union([position, 0.1*ift.from_random(operators["logdiffuse"].domain)])

    def get_mini(iglobal):
        if iglobal < 5:
            return ift.VL_BFGS(ift.GradientNormController(name="kl", iteration_limit=15))
        return ift.NewtonCG(ift.GradientNormController(name="kl", iteration_limit=15))

    def get_sampling(iglobal):
        return ift.AbsDeltaEnergyController(deltaE=0.5, convergence_level=3, iteration_limit=100,
                                            name="Sampling")

    def get_cst(iglobal):
        res = []
        if iglobal < 7:
            res += list(weightop.domain.keys())
            if enable_points:
                res += list(operators["points"].domain.keys())
        return res

    ift.optimize_kl(full_lh, 35, 6, get_mini, get_sampling, None,
                    constants=get_cst, point_estimates=get_cst,
                    initial_position=position, comm=rve.mpi.comm,
                    output_directory=cfg["output"]["directory"], **common)


if __name__ == "__main__":
    main(sys.argv[1])
