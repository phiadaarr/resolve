# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import configparser
from functools import reduce
from operator import add
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
    sky, operators = rve.multi_frequency_sky(cfg["sky"])
    raw_sky = operators["sky"]
    enable_points = operators["points"] is not None
    operators["logsky"] = raw_sky.log()
    n_imaging_bands = sky.target[2].size

    p = ift.Plot()
    for _ in range(9):
        p.add(raw_sky(ift.from_random(raw_sky.domain)), norm=LogNorm())
    if master:
        p.output(name="sky_prior_samples.png")
    # /Sky model

    # Bayesian weighting
    if enable_weighting:
        assert obs.npol == 1
        subcfg = cfg["weighting"]
        if subcfg["model"] == "cfm":
            # TODO Rename weight -> sigma_correction
            import ducc0

            npix = subcfg.getint("npix")
            fac = subcfg.getfloat("zeropadding factor")
            npix_padded = ducc0.fft.good_size(int(np.round(npix*fac)))

            xs = np.log(obs.effective_uvwlen().val)
            xs -= np.min(xs)

            maxlen = max(rve.mpi.comm.allgather(np.max(xs)))
            dom = ift.RGSpace(npix_padded, fac * maxlen / npix)

            cfm = ift.CorrelatedFieldMaker(prefix="weighting", total_N=n_imaging_bands)
            from resolve.sky_model import _parse_or_none
            kwargs = {kk: _parse_or_none(subcfg, f"{kk}") for kk in ["fluctuations", "loglogavgslope", "flexibility", "asperity"]}
            cfm.add_fluctuations(dom, **kwargs)
            kwargs = {
                "offset_mean": subcfg.getfloat("zero mode offset"),
                "offset_std": (subcfg.getfloat("zero mode mean"),
                               subcfg.getfloat("zero mode stddev"))
            }
            cfm.set_amplitude_total_offset(**kwargs)
            log_weights = cfm.finalize(0)

            mfs = rve.MultiFieldStacker(log_weights.target, 0, [str(ii) for ii in range(sky.target[2].size)])
            mfs1 = rve.MultiFieldStacker(obs.vis.domain[1:], 1, [str(ii) for ii in range(sky.target[2].size)])
            assert obs.npol == 1
            op = []
            for ii in range(sky.target[2].size):
                foo = ift.LinearInterpolator(dom, xs[0, :, ii][None])
                op.append(foo.ducktape(str(ii)).ducktape_left(str(ii)))
            log_weights = (mfs1 @ reduce(add, op) @ mfs.inverse @ log_weights).ducktape_left(obs.vis.domain)

            weightop = ift.makeOp(obs.weight) @ log_weights.scale(-2).exp()
            #operators["log_sigma_correction"] = log_weights
            #operators["sigma_correction"] = log_weights.exp()
            #operators["log_sigma_correction power spectrum"] = cfm.power_spectrum
        else:
            raise NotImplementedError
    else:
        weightop = None
    # /Bayesian weighting

    #lh = mf.MpiLikelihood(sky, obs, global_freqs, mpi_lo_hi, learnable_weights=log_weights,
                          #eff_uv_lengths=log_eff_uv_lengths)

    full_lh = rve.ImagingLikelihood(obs, sky, inverse_covariance_operator=weightop)
    lh_no_weights = rve.ImagingLikelihood(obs, sky)
    position = 0.1 * ift.from_random(lh_no_weights.domain)
    print(rve.profile_function(lh_no_weights, position, 1))

    del operators["sky"]
    del operators["logsky"]
    del operators["points"]
    print(operators.keys())
    common = {"plottable_operators": operators, "overwrite": True, "return_final_position": True}

    def get_mini(iglobal):
        if iglobal == 0:
            return ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=20))
        return ift.VL_BFGS(ift.GradientNormController(name="bfgs", iteration_limit=20))

    def get_sampling(iglobal):
        if iglobal == 0:
            return None
        return ift.AbsDeltaEnergyController(deltaE=0.5, convergence_level=3, iteration_limit=500,
                                            name="Sampling")

    def callback(sl, iglobal):
        if iglobal == 0:
            lh = lh_no_weights
        else:
            lh = full_lh
        s = ift.extra.minisanity(lh.data, lh.metric_at_pos, lh.model_data, sl)
        if rve.mpi.master:
            print(s)

    _, position = ift.optimize_kl(lh_no_weights, 1, 0, get_mini, get_sampling, None,
                                  #constants=get_cst, point_estimates=get_cst,
                                  initial_position=position, callback=callback,
                                  output_directory=cfg["output"]["directory"],
                                  **common)


if __name__ == "__main__":
    main(sys.argv[1])
