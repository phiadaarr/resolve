# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift


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
    res["global_iterations"] = total_iterations
    f_int = lambda s: _comma_separated_str_to_list(s, total_iterations, output_type=int)
    f = lambda s: _comma_separated_str_to_list(s, total_iterations)
    fnone = lambda s: _comma_separated_str_to_list(s, total_iterations, allow_none=True)
    sampling_iterations = f_int(cfg["sampling iteration limit"])
    res["n_samples"] = lambda ii: f_int(cfg["n samples"])[ii]
    res["sampling_iteration_controller"] = lambda ii: ift.AbsDeltaEnergyController(0.05, iteration_limit=sampling_iterations[ii], convergence_level=3, name="Sampling")
    res["output_directory"] = cfg["output folder"]


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
                                              output_type=int)[ii]
        ic = ift.AbsDeltaEnergyController(.001, iteration_limit=lim, name=f"iter{ii} {f(cfg['nonlinear sampling optimizer'])[ii]}",
                                          convergence_level=3)
        return opt(ic)

    res["kl_minimizer"] = optimizer
    res["nonlinear_sampling_minimizer"] = nonlinear_sampling

    lhlst = _comma_separated_str_to_list(cfg["likelihood"], total_iterations)
    res["likelihood_energy"] = lambda ii: likelihood_dct[lhlst[ii]]

    cstlst = _comma_separated_str_to_list(cfg["constants"], total_iterations, allow_none=True)
    constants_dct[None] = ift.MultiDomain.make({})
    res["point_estimates"] = res["constants"] = lambda ii: constants_dct[cstlst[ii]].keys()

    return res


def _comma_separated_str_to_list(cfg, length, allow_none=False, output_type=None):
    # TODO Implement basic arithmetics (e.g. 4*NewtonCG)
    lst = cfg.split(",")
    lst = list(map(_nonestr_to_none, lst))

    if len(lst) == 1:
        lst = length * lst
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

    return lst


def _nonestr_to_none(s):
    if s.lower() in ["none", ""]:
        return None
    return s


def _to_type(obj, output_type):
    if obj is None:
        return None
    return output_type(obj)
