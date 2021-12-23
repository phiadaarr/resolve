# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import configparser
import os
import sys

import nifty8 as ift

import resolve as rve
from resolve.mpi import master


def main(cfg_file_name):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file_name)

    obs_calib_flux, obs_calib_phase, obs_science = rve.parse_data_config(cfg)

    if cfg["sky"]["polarization"] == "I":
        obs_science = [oo.restrict_to_stokesi().average_stokesi() for oo in obs_science]
    assert len(obs_calib_flux) == len(obs_calib_phase) == 0

    # Model operators
    sky, ops_sky, domains = rve.sky_model(cfg["sky"], obs_science)
    enable_points = "points" in domains
    assert len(obs_science) == 1
    weights, ops_weights = rve.weighting_model(cfg["weighting"], obs_science[0], sky.target)
    # FIXME Add plots for weights
    operators = {**ops_sky, **ops_weights}
    domains["weights"] = weights.domain
    # /Model operators

    # Likelihoods
    assert len(obs_science) == 1
    lhs = {}
    lhs["full"] = rve.ImagingLikelihood(obs_science[0], sky, inverse_covariance_operator=weights)
    if enable_points:
        lhs["points"] = rve.ImagingLikelihood(obs_science[0], operators["points"])
    lhs["data weights"] = rve.ImagingLikelihood(obs_science[0], sky)
    # /Likelihoods

    # Initial position
    position = 0.1 * ift.from_random(lhs["full"].domain)
    rve.mpi.barrier(rve.mpi.comm)
    s = rve.profile_function(lhs["full"], position, 1)
    if master:
        print(s)
    del position
    rve.mpi.barrier(rve.mpi.comm)
    # /Initial position

    # Assumption: likelihood is not MPI distributed
    get_comm = rve.mpi.comm
    ift.optimize_kl(**rve.parse_optimize_kl_config(cfg["optimization"], lhs, domains),
                    plottable_operators=operators, comm=get_comm, overwrite=True)


if __name__ == "__main__":
    main(sys.argv[1])
