# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras, Jakob Knollmüller

import numpy as np
import configparser
import os
import sys
from distutils.util import strtobool

import nifty8 as ift
import resolve as rve
from matplotlib.colors import LogNorm
from resolve.mpi import master


def main(cfg_file_name):
    cfg = configparser.ConfigParser()
    cfg.read(cfg_file_name)
    if master:
        print(f"Load {cfg_file_name}")

    point_sources = [("0deg", "0deg")]

    rve.set_epsilon(cfg["response"].getfloat("epsilon"))
    rve.set_wgridding(strtobool(cfg["response"]["wgridding"]))
    rve.set_double_precision(True)
    rve.set_nthreads(cfg["technical"].getint("nthreads"))

    nx = cfg["sky"].getint("space npix x")
    ny = cfg["sky"].getint("space npix y")
    dx = rve.str2rad(cfg["sky"]["space fov x"]) / nx
    dy = rve.str2rad(cfg["sky"]["space fov y"]) / ny
    dom = ift.RGSpace([nx, ny], [dx, dy])

    # Sky model
    # Diffuse
    cfm_zm_args = {
        "offset_mean":
            cfg["sky"].getfloat("zero mode offset"),
        "offset_std": (
            cfg["sky"].getfloat("zero mode mean"),
            cfg["sky"].getfloat("zero mode stddev"),
        ),
    }
    cfm_spatial_args = {
        "fluctuations": (
            cfg["sky"].getfloat("space fluctuations mean"),
            cfg["sky"].getfloat("space fluctuations stddev"),
        ),
        "loglogavgslope": (
            cfg["sky"].getfloat("space loglogavgslope mean"),
            cfg["sky"].getfloat("space loglogavgslope stddev"),
        ),
        "flexibility": (
            cfg["sky"].getfloat("space flexibility mean"),
            cfg["sky"].getfloat("space flexibility stddev"),
        ),
        "asperity": (
            cfg["sky"].getfloat("space asperity mean"),
            cfg["sky"].getfloat("space asperity stddev"),
        ),
    }
    logdiffuse = ift.SimpleCorrelatedField(dom, **cfm_spatial_args, **cfm_zm_args)

    # Point sources
    ppos = []
    for point in point_sources:
        ppos.append([rve.str2rad(point[0]), rve.str2rad(point[1])])
    inserter = rve.PointInserter(dom, ppos)
    points = ift.InverseGammaOperator(
        inserter.domain, alpha=0.5, q=0.2 / dom.scalar_dvol).ducktape("points")
    points = inserter @ points

    sky = logdiffuse.exp() + points
    # /Sky model

    p = ift.Plot()
    for _ in range(9):
        p.add(sky(ift.from_random(sky.domain)), norm=LogNorm())
    if master:
        p.output(name="sky_prior_samples.png")

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

    # Weightop
    npix = 2500
    effuv = obs.effective_uvwlen().val[0]
    assert obs.nfreq == obs.npol == 1
    dom = ift.RGSpace(npix, 2 * np.max(effuv) / npix)
    logwgt = ift.SimpleCorrelatedField(dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5),
                                       "invcov")
    li = ift.LinearInterpolator(dom, effuv.T)
    conv = rve.DomainChangerAndReshaper(li.target, obs.weight.domain)
    weightop = ift.makeOp(obs.weight) @ (conv @ li @ logwgt.exp()) ** (-2)
    # /Weightop

    full_lh = rve.ImagingLikelihood(obs, sky, inverse_covariance_operator=weightop)
    position = 0.1 * ift.from_random(full_lh.domain)

    common = {
        "output_directory": cfg["output"]["directory"],
        "plottable_operators": {
            "sky": sky,
            "logsky": sky.log(),
            "points": points,
            "logdiffuse": logdiffuse,
            "logdiffuse pspec": logdiffuse.power_spectrum
        },
        "overwrite": True
    }

    if master:
        # Points only, MAP
        lh = rve.ImagingLikelihood(obs, points)
        sl = ift.optimize_kl(
            lh,
            1,
            0,
            ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=4)),
            None,
            None,
            initial_position=position,
            **common)
        position = ift.MultiField.union([position, sl.local_item(0)])

        # Points constant, diffuse only, MAP
        lh = rve.ImagingLikelihood(obs, sky)
        sl = ift.optimize_kl(
            lh,
            1,
            0,
            ift.NewtonCG(ift.GradientNormController(name="hamiltonian", iteration_limit=20)),
            None,
            None,
            initial_position=position,
            initial_index=1,
            **common)
        position = ift.MultiField.union([position, sl.local_item(0)])

        cst = sky.domain.keys()
        sl = ift.optimize_kl(
            full_lh,
            1,
            0,
            ift.VL_BFGS(ift.GradientNormController(name="bfgs", iteration_limit=20)),
            ift.AbsDeltaEnergyController(0.1, 3, 100, name="Sampling"),
            None,
            constants=cst,
            point_estimates=cst,
            initial_position=position,
            initial_index=2,
            **common)
        position = ift.MultiField.union([position, sl.local_item(0)])

    if rve.mpi.mpi:
        if not master:
            position = None
        position = rve.mpi.comm.bcast(position, root=0)
    ift.random.push_sseq_from_seed(42)

    # First MGVI diffuse sky, then sky + weighting simultaneously

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
        return ift.AbsDeltaEnergyController(deltaE=0.5, iteration_limit=lim)

    def get_cst(iglobal):
        if iglobal < 7:
            return list(points.domain.keys()) + list(weightop.domain.keys())
        else:
            return []

    sl = ift.optimize_kl(
        full_lh,
        35,
        5,
        get_mini,
        get_sampling,
        None,
        constants=get_cst,
        point_estimates=get_cst,
        initial_position=position,
        initial_index=7,
        **common)


if __name__ == "__main__":
    main(sys.argv[1])
