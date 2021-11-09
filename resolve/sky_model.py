# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift

from .constants import str2rad
from .points import PointInserter


def single_frequency_sky(cfg_section, list_point_sources=[]):
    nx = cfg_section.getint("space npix x")
    ny = cfg_section.getint("space npix y")
    dx = str2rad(cfg_section["space fov x"]) / nx
    dy = str2rad(cfg_section["space fov y"]) / ny
    dom = ift.RGSpace([nx, ny], [dx, dy])

    logdiffuse, cfm = cfm_from_cfg(cfg_section, {"space": dom}, "sky diffuse" )
    sky = logdiffuse.exp()
    add = {"logdiffuse": logdiffuse, "logdiffuse power spectrum": cfm.power_spectrum}

    # Point sources
    mode = cfg_section["point sources mode"]
    if mode == "single":
        ppos = []
        s = cfg_section["point sources locations"]
        for xy in s.split(";"):
            x, y = xy.split(",")
            ppos.append((str2rad(x), str2rad(y)))
        inserter = PointInserter(dom, ppos)
        alpha = cfg_section.getfloat("point sources alpha")
        q = cfg_section.getfloat("point sources q")
        points = ift.InverseGammaOperator(inserter.domain, alpha=alpha, q=q/dom.scalar_dvol)
        points = inserter @ points.ducktape("points")
        sky = sky + points
        add["points"] = points

    elif mode == "disable":
        add["points"] = None
    else:
        raise ValueError(f"In order to disable point source component, set `point sources mode` to `disable`. Got: {mode}")
    return sky, add


def cfm_from_cfg(cfg_section, domain_dct, prefix):
    if not isinstance(domain_dct, dict):
        domain_dct = {"": domain_dct}
    cfm = ift.CorrelatedFieldMaker(_append_to_nonempty_string(prefix, " "))
    for key_prefix, dom in domain_dct.items():
        foo = _append_to_nonempty_string(key_prefix, " ")
        kwargs = {kk: tuple(cfg_section.getfloat(f"{foo}{kk} {stat}") for stat in ["mean", "stddev"]) for kk in ["fluctuations", "loglogavgslope", "flexibility", "asperity"]}
        kwargs["prefix"] = key_prefix
        cfm.add_fluctuations(dom, **kwargs)
    kwargs = {
        "offset_mean": cfg_section.getfloat("zero mode offset"),
        "offset_std": (cfg_section.getfloat("zero mode mean"),
                       cfg_section.getfloat("zero mode stddev"))
    }
    cfm.set_amplitude_total_offset(**kwargs)
    op = cfm.finalize(prior_info=0)
    return op, cfm


def _append_to_nonempty_string(s, append):
    if s == "":
        return s
    return s + append
