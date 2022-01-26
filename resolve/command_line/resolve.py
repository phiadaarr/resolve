# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import configparser
import os
import sys

import nifty8 as ift

from ..config_utils import parse_data_config, parse_optimize_kl_config
from ..global_config import set_verbosity
from ..likelihood import ImagingLikelihood
from ..mpi import barrier, comm, master
from ..sky_model import sky_model
from ..util import profile_function
from ..weighting_model import weighting_model


def main():
    cfg_file_name = sys.argv[1]
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file_name)

    obs_calib_flux, obs_calib_phase, obs_science = parse_data_config(cfg)

    if cfg["sky"]["polarization"] == "I":
        obs_science = [oo.restrict_to_stokesi().average_stokesi() for oo in obs_science]
    assert len(obs_calib_flux) == len(obs_calib_phase) == 0

    # Model operators
    sky, ops_sky, domains = sky_model(cfg["sky"], obs_science)
    enable_points = "points" in domains
    weights, ops_weights = weighting_model(cfg["weighting"], obs_science, sky.target)
    operators = {**ops_sky, **ops_weights}
    domains["weights"] = ift.MultiDomain.union([ww.domain for ww in weights])
    # /Model operators

    # Likelihoods
    lhs = {}
    lhs["full"] = ImagingLikelihood(obs_science, sky, inverse_covariance_operator=weights)
    if enable_points:
        lhs["points"] = ImagingLikelihood(obs_science, operators["points"])
    lhs["data weights"] = ImagingLikelihood(obs_science, sky)
    # /Likelihoods

    # Profiling
    position = 0.1 * ift.from_random(lhs["full"].domain)
    set_verbosity(True)
    barrier(comm)
    s = profile_function(lhs["full"], position, 1)
    if master:
        print(s)
    del position
    barrier(comm)
    set_verbosity(False)
    # /Profiling

    # Assumption: likelihood is not MPI distributed
    get_comm = comm
    if "points" in operators:
        del operators["points"]
    ift.optimize_kl(**parse_optimize_kl_config(cfg["optimization"], lhs, domains),
                    plottable_operators=operators, comm=get_comm, overwrite=True,
                    plot_latent=True)


if __name__ == "__main__":
    main()
