# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import configparser
import os
import sys
from distutils.util import strtobool

import nifty8 as ift
import resolve as rve
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

    # Model operators
    sky, ops_sky, keys = rve.sky_model(cfg["sky"], data_freq=obs.freq)
    enable_points = len(keys["points"]) > 0
    weights, ops_weights = rve.weighting_model(cfg["weighting"], obs, sky.target)
    # FIXME Add plots for weights
    operators = {**ops_sky, **ops_weights}
    keys["weights"] = weights.domain.keys()
    keys["sky"] = sky.domain.keys()

    for kk in keys:
        keys[kk] = tuple(keys[kk])
    # /Model operators

    # Likelihoods
    full_lh = rve.ImagingLikelihood(obs, sky, inverse_covariance_operator=weights)
    if enable_points:
        point_lh = rve.ImagingLikelihood(obs, operators["points"])
    sky_lh = rve.ImagingLikelihood(obs, sky)
    # /Likelihoods

    # Initial position
    position = 0.1 * ift.from_random(full_lh.domain)
    s = rve.profile_function(full_lh, position, 1)
    if master:
        print(s)
    # /Initial position

    # Optimization
    common = {"plottable_operators": operators, "overwrite": True}

    def get_mini(iglobal):
        if iglobal == 0:
            return ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=4))
        if iglobal == 1:
            return ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=20))
        if iglobal < 7:
            return ift.VL_BFGS(ift.GradientNormController(name="bfgs", iteration_limit=20))
        if iglobal < 12:
            return ift.VL_BFGS(ift.GradientNormController(name="kl", iteration_limit=15))
        return ift.NewtonCG(ift.GradientNormController(name="kl", iteration_limit=15))

    def get_sampling(iglobal):
        if iglobal in [0, 1]:
            return None
        return ift.AbsDeltaEnergyController(deltaE=0.5, convergence_level=3, iteration_limit=500,
                                            name="Sampling")

    def get_cst(iglobal):
        if iglobal in [0, 1]:
            return []
        if iglobal < 7:
            return keys["sky"]
        if iglobal < 14:
            return keys["weights"] + keys["points"]
        return []

    def get_lh(iglobal):
        if iglobal == 0:
            return point_lh
        if iglobal == 1:
            return sky_lh
        return full_lh

    def get_n_samples(iglobal):
        if iglobal in [0, 1]:
            return 0
        if iglobal < 7:
            return 4
        return 4

    def get_comm(iglobal):
        return rve.mpi.comm

    def callback(sl, iglobal, position):
        lh = get_lh(iglobal)
        s = ift.extra.minisanity(lh.data, lh.metric_at_pos, lh.model_data, sl)
        if rve.mpi.master:
            print(s)
        # Reset diffuse component
        if iglobal == 6:
            diffuse_domain = {kk: vv for kk, vv in full_lh.domain.items() if kk in keys["diffuse"]}
            return ift.MultiField.union([position, 0.1*ift.from_random(diffuse_domain)])

    ift.optimize_kl(get_lh, 40, get_n_samples, get_mini, get_sampling, None,
                    constants=get_cst, point_estimates=get_cst,
                    initial_index=0 if enable_points else 1,
                    initial_position=position, comm=get_comm, callback=callback,
                    output_directory=cfg["output"]["directory"],
                    **common)
    # /Optimization


if __name__ == "__main__":
    main(sys.argv[1])
