# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021-2022 Max-Planck-Society
# Author: Philipp Arras

import os
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
from functools import reduce
from operator import add

import nifty8 as ift

from ..config_utils import parse_data_config, parse_optimize_kl_config
from ..global_config import set_verbosity, set_nthreads
from ..likelihood import ImagingLikelihood
from ..mpi import barrier, comm, master
from ..sky_model import sky_model_diffuse, sky_model_points
from ..util import profile_function
from ..weighting_model import weighting_model, visualize_weights


def main():
    parser = ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("-j", type=int, default=1,
                        help="Number of threads for thread parallelization")
    args = parser.parse_args()

    cfg = ConfigParser()
    cfg.read(args.config_file)
    set_nthreads(args.j)

    obs_calib_flux, obs_calib_phase, obs_science = parse_data_config(cfg)

    if cfg["sky"]["polarization"] == "I":
        obs_science = [oo.restrict_to_stokesi().average_stokesi() for oo in obs_science]
    assert len(obs_calib_flux) == len(obs_calib_phase) == 0

    # Model operators
    diffuse, additional_diffuse = sky_model_diffuse(cfg["sky"], obs_science)
    points, additional_points = sky_model_points(cfg["sky"], obs_science)
    sky = reduce(add, (op for op in [diffuse, points] if op is not None))
    weights, additional_weights = weighting_model(cfg["weighting"], obs_science, sky.target)
    operators = {**additional_diffuse, **additional_points, **additional_weights}
    operators["sky"] = sky
    # /Model operators

    # Domains
    domains = {}
    if diffuse is not None:
        domains["diffuse"] = diffuse.domain
    if points is not None:
        domains["points"] = points.domain
    if weights is not None:
        domains["weights"] = ift.MultiDomain.union([ww.domain for ww in weights])
    domains["sky"] = sky.domain
    # /Domains

    # Likelihoods
    lhs = {}
    lhs["full"] = ImagingLikelihood(obs_science, sky, inverse_covariance_operator=weights)
    if points is not None:
        lhs["points"] = ImagingLikelihood(obs_science, points)
    lhs["data weights"] = ImagingLikelihood(obs_science, sky)
    # /Likelihoods

    outdir = parse_optimize_kl_config(cfg["optimization"], lhs, domains)["output_directory"]

    # Profiling
    position = 0.1 * ift.from_random(lhs["full"].domain)
    set_verbosity(True)
    barrier(comm)
    s = "\n\nProfile likelihood\n" + profile_function(lhs["full"], position, 1)
    if master:
        # FIXME Use python's logger module for this
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "log.txt"), "a") as f:
            f.write(s)
        print(s)
    del position
    barrier(comm)
    set_verbosity(False)
    # /Profiling

    def inspect_callback(sl, iglobal, position):
        visualize_weights(obs_science, sl, iglobal, sky, weights, outdir, io=master)

    # Assumption: likelihood is not MPI distributed
    get_comm = comm
    ift.optimize_kl(**parse_optimize_kl_config(cfg["optimization"], lhs, domains, inspect_callback),
                    plottable_operators=operators, comm=get_comm, overwrite=True,
                    plot_latent=True)


if __name__ == "__main__":
    main()
