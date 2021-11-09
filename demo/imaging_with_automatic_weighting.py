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
    enable_points = operators["points"] is not None
    operators["sky"] = sky
    operators["logsky"] = sky.log()

    p = ift.Plot()
    for _ in range(9):
        p.add(sky(ift.from_random(sky.domain)), norm=LogNorm())
    if master:
        p.output(name="sky_prior_samples.png")
    # /Sky model

    # Bayesian weighting
    if enable_weighting:
        assert obs.nfreq == obs.npol == 1
        subcfg = cfg["weighting"]
        if subcfg["model"] == "cfm":
            npix = 2500  # FIXME Use numbers from config file
            xs = np.log(obs.effective_uvwlen().val)
            xs -= np.min(xs)
            if not xs.shape[0] == xs.shape[2] == 1:
                raise RuntimeError
            # FIXME Use Integrated Wiener process
            dom = ift.RGSpace(npix, 2 * np.max(xs) / npix)
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
    common = {"output_directory": cfg["output"]["directory"],
              "plottable_operators": operators,
              "overwrite": True}

    if master:
        common["comm"] = rve.mpi.comm_self
        if enable_points:
            # Points only, MAP
            lh = rve.ImagingLikelihood(obs, operators["points"])
            mini = ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=4))
            sl = ift.optimize_kl(lh, 1, 0, mini, None, None, initial_position=position, **common)
            position = sl.local_item(0)

        # Points constant, diffuse only, MAP
        lh = rve.ImagingLikelihood(obs, sky)
        mini = ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=20))
        sl = ift.optimize_kl(lh, 1, 0, mini, None, None, initial_position=position, initial_index=1, **common)
        position = sl.local_item(0)

        # Only weighting, MGVI
        cst = sky.domain.keys()
        mini = ift.VL_BFGS(ift.GradientNormController(name="bfgs", iteration_limit=20))
        ic_sampling = ift.AbsDeltaEnergyController(0.1, 3, 100, name="Sampling")
        sl = ift.optimize_kl(full_lh, 5, 3, mini, ic_sampling, None,
                             constants=cst, point_estimates=cst,
                             initial_position=position, initial_index=2, **common)
        position = sl.local_item(0)

        # Reset sky
        position = ift.MultiField.union([0.1*ift.from_random(sky.domain), position.extract(weightop.domain)])

    if rve.mpi.mpi:
        rve.mpi.comm.Barrier()
        if not master:
            position = None
        position = rve.mpi.comm.bcast(position, root=0)
    ift.random.push_sseq_from_seed(42)
    common["comm"] = rve.mpi.comm

    def get_mini(iglobal):
        if iglobal < 7:
            return ift.NewtonCG(ift.GradientNormController(name="kl", iteration_limit=15))
        elif iglobal < 7 + 5:
            return ift.VL_BFGS(ift.GradientNormController(name="kl", iteration_limit=15))
        else:
            return ift.NewtonCG(ift.GradientNormController(name="kl", iteration_limit=15))

    def get_sampling(iglobal):
        if iglobal < 7:
            lim = 200
        elif iglobal < 7 + 5:
            lim = 700
        else:
            lim = 700
        return ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=lim, name="Sampling")

    def get_cst(iglobal):
        res = []
        if iglobal < 7:
            res += list(weightop.domain.keys())
            if enable_points:
                res += list(operators["points"].domain.keys())
        return res

    # First MGVI diffuse sky, then sky + weighting simultaneously
    sl = ift.optimize_kl(full_lh, 35, 6, get_mini, get_sampling, None,
                         constants=get_cst, point_estimates=get_cst,
                         initial_position=position, initial_index=7, **common)


if __name__ == "__main__":
    main(sys.argv[1])
