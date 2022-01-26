# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import os
from datetime import datetime
from distutils.util import strtobool
from glob import glob

import nifty8 as ift
import numpy as np

from .data.ms_import import ms2observations
from .data.observation import Observation
from .global_config import (set_double_precision, set_epsilon, set_nthreads,
                            set_wgridding)
from .mpi import master


def parse_data_config(cfg):
    set_epsilon(cfg["response"].getfloat("epsilon"))
    set_wgridding(strtobool(cfg["response"]["wgridding"]))
    set_double_precision(cfg["response"].getboolean("double precision"))
    set_nthreads(cfg["response"].getint("nthreads"))

    obs_calib_phase = _cfg_to_observations(cfg["data"]["phase calibrator"])
    obs_calib_flux = _cfg_to_observations(cfg["data"]["flux calibrator"])
    obs_science = _cfg_to_observations(cfg["data"]["science target"])

    s = "number of randomly sampled rows"
    comm = ift.utilities.get_MPI_params()[0]
    if s in cfg["data"]:
        nvis = cfg["data"].getint(s)

        ift.utilities.check_MPI_synced_random_state(comm)
        for oo in [obs_calib_phase, obs_calib_flux, obs_science]:
            for ii in range(len(oo)):
                inds = ift.random.current_rng().choice(np.arange(obs_science[0].nrow), (nvis,),
                                                       replace=False)
                oo[ii] = oo[ii][inds]
        assert all(oo.vis.shape[1] == nvis for oo in obs_science)
        ift.utilities.check_MPI_synced_random_state(comm)

    return obs_calib_flux, obs_calib_phase, obs_science


def parse_optimize_kl_config(cfg, likelihood_dct, constants_dct={}):
    """
    Parameters
    ----------
    cfg : 

    likelihood_dct : dict
        Dictionary of likelihood operators that is
        used to look up the likehoods.
    constants_dct : dict
        Dictionary of domains that is used to look up the domains for partially
        constant optimization.

    Returns
    -------
    dict : that can be plugged into `ift.optimize_kl`.
    """

    res = {}

    total_iterations = cfg.getint("total iterations")
    res["total_iterations"] = total_iterations
    f_int = lambda s: _comma_separated_str_to_list(s, total_iterations, output_type=int)
    f = lambda s: _comma_separated_str_to_list(s, total_iterations)
    fnone = lambda s: _comma_separated_str_to_list(s, total_iterations, allow_none=True)
    sampling_iterations = f_int(cfg["sampling iteration limit"])
    res["n_samples"] = lambda ii: f_int(cfg["n samples"])[ii]
    res["sampling_iteration_controller"] = lambda ii: ift.AbsDeltaEnergyController(0.05, iteration_limit=sampling_iterations[ii], convergence_level=3, name="Sampling")
    res["output_directory"] = os.path.expanduser(cfg["output folder"])


    def optimizer(ii):
        opt = getattr(ift, _comma_separated_str_to_list(cfg["optimizer"], total_iterations)[ii])
        lim = _comma_separated_str_to_list(cfg["optimizer iteration limit"], total_iterations,
                                              output_type=int)[ii]
        ic = ift.AbsDeltaEnergyController(.001, iteration_limit=lim, name=f"iter{ii} {f(cfg['optimizer'])[ii]}",
                                          convergence_level=3)
        return opt(ic)


    def nonlinear_sampling(ii):
        name = _comma_separated_str_to_list(cfg["nonlinear sampling optimizer"], total_iterations, allow_none=True)[ii]
        if name is None:
            return None
        opt = getattr(ift, name)
        lim = _comma_separated_str_to_list(cfg["nonlinear sampling optimizer iteration limit"], total_iterations,
                                           output_type=int, allow_none=True)[ii]
        if lim is None:
            raise ValueError()
        ic = ift.AbsDeltaEnergyController(.001, iteration_limit=lim,
                                          name=f"iter{ii} {name}",
                                          convergence_level=3)
        return opt(ic)

    res["kl_minimizer"] = optimizer
    res["nonlinear_sampling_minimizer"] = nonlinear_sampling

    lhlst = _comma_separated_str_to_list(cfg["likelihood"], total_iterations)
    res["likelihood_energy"] = lambda ii: likelihood_dct[lhlst[ii]]

    reset = _comma_separated_str_to_list(cfg["reset"], total_iterations, allow_none=True)

    def callback(sl, iglobal, position):
        lh = res["likelihood_energy"](iglobal)
        s = "\n".join(
            ["", "",
             f"Finished index: {iglobal}",
             f"Current datetime: {datetime.now()}",
             ift.extra.minisanity(lh.data, lh.metric_at_pos, lh.model_data, sl,
                                  terminal_colors=False),
             ""]
            )
        if master:
            # FIXME Use python's logger module for this
            with open(os.path.join(res["output_directory"], "log.txt"), "a") as f:
                f.write(s)
            print(s)

        if reset[iglobal] is not None:
            reset_domain = constants_dct[reset[iglobal]]
            return ift.MultiField.union([position, 0.1*ift.from_random(reset_domain)])

    res["inspect_callback"] = callback
    constants_dct[None] = ift.MultiDomain.make({})

    cstlst = _comma_separated_str_to_list(cfg["point estimates"], total_iterations, allow_none=True)
    res["point_estimates"] = lambda ii: constants_dct[cstlst[ii]].keys()

    cstlst = _comma_separated_str_to_list(cfg["constants"], total_iterations, allow_none=True)
    res["constants"] = lambda ii: constants_dct[cstlst[ii]].keys()

    res["resume"] = cfg.getboolean("resume")
    res["save_strategy"] = cfg.get("save strategy")

    return res


def _cfg_to_observations(cfg):
    newcfg = []
    for cc in cfg.split(","):
        if len(cc) == 0:
            continue
        fname, options = cc.split(":")
        for ff in glob(os.path.expanduser(fname)):
            newcfg.append(f"{ff}:{options}")
    cfg = ",".join(newcfg)

    res = []
    for cc in cfg.split(","):
        cc = cc.split(":")
        file_name = cc.pop(0)
        if file_name == "":
            continue
        ext = os.path.splitext(file_name)[1]
        if ext == ".npz":
            assert cc == ['']
            obs = Observation.load(file_name)
        elif ext == ".ms":  # ms
            assert len(cc) == 1
            cc = cc[0]
            assert cc[0] == "("
            assert cc[-1] == ")"
            cc = cc[1:-1]
            source, spectral_window, data_column = cc.split("$")
            obs = ms2observations(file_name, data_column, True, int(spectral_window), field=source)
            assert len(obs) == 1
            obs = obs[0]
        else:
            raise RuntimeError("Paths with ':' and ',' are not allowed")
        res.append(obs)
    return res


def _comma_separated_str_to_list(cfg, length, allow_none=False, output_type=None):
    # TODO Implement basic arithmetics (e.g. 4*NewtonCG)
    lst = cfg.split(",")
    lst = list(map(_nonestr_to_none, lst))

    if len(lst) == 1:
        lst = length * lst
    # Parse multiplication
    tmp = []
    for ii, ll in enumerate(lst):
        if ll is not None and len(ll) > 1 and "*" in ll:
            ind = ll.index("*")
            tmp.extend(int(ll[:ind]) * [ll[ind+1:]])
        else:
            tmp.append(ll)
    lst = tmp
    lst = list(map(_nonestr_to_none, lst))
    # /Parse multiplication

    # Parse *
    if lst.count("*") > 1:
        raise ValueError("Only one * allowed")
    if len(lst) != length:
        ind = lst.index("*")
        if ind == 0:
            raise ValueError("* at beginning not allowed")
        lst.pop(ind)
        for _ in range(length - len(lst)):
            lst.insert(ind - 1, lst[ind - 1])
    # /Parse *

    if None in lst and not allow_none:
        raise ValueError("None is not allowed")

    if output_type is not None:
        lst = list(map(lambda x: _to_type(x, output_type), lst))

    assert len(lst) == length
    return lst


def _nonestr_to_none(s):
    if s is None or s.lower() in ["none", ""]:
        return None
    return s


def _to_type(obj, output_type):
    if obj is None:
        return None
    return output_type(obj)
