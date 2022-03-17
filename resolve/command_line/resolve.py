# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import os
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
from functools import reduce
from operator import add

import nifty8 as ift

from ..config_utils import parse_data_config, parse_optimize_kl_config
from ..likelihood import ImagingLikelihood
from ..mpi import barrier, comm, master
from ..plot.baseline_histogram import visualize_weighted_residuals
from ..sky_model import sky_model_diffuse, sky_model_points
from ..weighting_model import log_weighting_model
from ..plot.sky import plot_sky


def main():
    parser = ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("-j", type=int, default=1,
                        help="Number of threads for thread parallelization")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--verbose", "-v", action="count", default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--terminate", type=int)
    args = parser.parse_args()

    nthreads = args.j
    ift.set_nthreads(nthreads)

    # Read config file
    if not os.path.isfile(args.config_file):
        raise RuntimeError(f"Config file {args.config_file} not found")
    output_directory = os.path.split(args.config_file)[0]
    if output_directory == "":
        output_directory = "."
    cfg = ConfigParser()
    cfg.read(args.config_file)
    # /Read config file

    # Read data
    obs_calib_flux, obs_calib_phase, obs_science = parse_data_config(cfg)

    if cfg["sky"]["polarization"] == "I":
        obs_science = [oo.restrict_to_stokesi().average_stokesi() for oo in obs_science]
    assert len(obs_calib_flux) == len(obs_calib_phase) == 0
    # /Read data

    # Model operators
    diffuse, additional_diffuse = sky_model_diffuse(cfg["sky"], obs_science, nthreads=nthreads)
    points, additional_points = sky_model_points(cfg["sky"], obs_science, nthreads=1)
    sky = reduce(add, (op for op in [diffuse, points] if op is not None))
    logweights, additional_weights = log_weighting_model(cfg["weighting"], obs_science, sky.target)
    if logweights is None:
        weights = None
    else:
        weights = [ww.exp() for ww in logweights]
    operators = {**additional_diffuse, **additional_points, **additional_weights}
    operators["sky"] = sky
    # /Model operators

    # Domains
    domains = {}
    if diffuse is not None:
        domains["diffuse"] = diffuse.domain
    if points is not None:
        domains["points"] = points.domain
    if logweights is not None:
        domains["weights"] = ift.MultiDomain.union([ww.domain for ww in logweights])
    domains["sky"] = sky.domain
    # /Domains

    # Likelihoods
    do_wgridding = cfg["response"].getboolean("wgridding")
    epsilon = cfg["response"].getfloat("epsilon")
    lhs = {}
    lhs["full"] = ImagingLikelihood(obs_science, sky, log_inverse_covariance_operator=logweights, epsilon=epsilon, do_wgridding=do_wgridding, nthreads=nthreads)
    if points is not None:
        lhs["points"] = ImagingLikelihood(obs_science, points, epsilon=epsilon, do_wgridding=do_wgridding, nthreads=nthreads)
    lhs["data weights"] = ImagingLikelihood(obs_science, sky, epsilon=epsilon, do_wgridding=do_wgridding, nthreads=nthreads)
    # /Likelihoods

    # Profiling
    position = 0.1 * ift.from_random(lhs["full"].domain)
    barrier(comm)
    if master:
        os.makedirs(output_directory, exist_ok=True)
        with ift.random.Context(12):
            ift.exec_time(lhs["full"], verbose=args.verbose)
    if args.profile_only:
        exit()
    del position
    barrier(comm)
    # /Profiling

    # Inference
    def inspect_callback(sl, iglobal, position):
        visualize_weighted_residuals(obs_science, sl, iglobal, sky, weights, output_directory, io=master,
                                     do_wgridding=do_wgridding, epsilon=epsilon, nthreads=nthreads)
        plot_sky(sl.average(sky), os.path.join(output_directory, f"sky/sky_{iglobal}.pdf"))

    if args.terminate is None:
        terminate_callback = lambda iglobal: False
    else:
        terminate_callback = lambda iglobal: iglobal == args.terminate

    # Assumption: likelihood is not MPI distributed
    get_comm = comm
    ift.optimize_kl(**parse_optimize_kl_config(cfg["optimization"], lhs, domains, inspect_callback),
                    plottable_operators=operators, comm=get_comm, overwrite=True,
                    plot_latent=True, resume=args.resume, terminate_callback=terminate_callback,
                    output_directory=output_directory)
    # /Inference


if __name__ == "__main__":
    main()
