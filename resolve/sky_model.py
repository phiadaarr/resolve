# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np

from .constants import str2rad
from .irg_space import IRGSpace
from .points import PointInserter
from .polarization_space import PolarizationSpace
from .simple_operators import DomainChangerAndReshaper
from .util import assert_sky_domain


def single_frequency_sky(cfg_section):
    dom = _spatial_dom(cfg_section)

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

    add["sky"] = sky
    sky = DomainChangerAndReshaper(sky.target, _default_sky_domain(sdom=dom)) @ sky
    assert_sky_domain(sky.target)
    return sky, add


def multi_frequency_sky(cfg):
    if "point sources mode" in cfg:
        raise NotImplementedError("Point sources are not supported yet.")

    sdom = _spatial_dom(cfg)

    if cfg["freq mode"] == "cfm":
        fnpix, df = cfg.getfloat("freq npix"), cfg.getfloat("freq pixel size")
        fdom = IRGSpace(cfg.getfloat("freq start") + np.arange(fnpix)*df)
        fdom_rg = ift.RGSpace(fnpix, df)
        logdiffuse, cfm = cfm_from_cfg(cfg, {"freq": fdom_rg, "space": sdom}, "sky diffuse")
        sky = logdiffuse.exp()
        add = {"logdiffuse": logdiffuse}
        fampl, sampl = list(cfm.get_normalized_amplitudes())
        add["freq normalized power spectrum"] = fampl**2
        add["space normalized power spectrum"] = sampl**2

        add["sky"] = sky
        reshape = DomainChangerAndReshaper(sky.target, _default_sky_domain(fdom=fdom, sdom=sdom))
        sky = reshape @ sky

    elif cfg["freq mode"] == "integrated wiener process":
        raise NotImplementedError
    else:
        raise RuntimeError

    assert_sky_domain(sky.target)
    return sky, add


def cfm_from_cfg(cfg_section, domain_dct, prefix):
    if not isinstance(domain_dct, dict):
        domain_dct = {"": domain_dct}
    cfm = ift.CorrelatedFieldMaker(_append_to_nonempty_string(prefix, " "))
    for key_prefix, dom in domain_dct.items():
        foo = _append_to_nonempty_string(key_prefix, " ")
        kwargs = {kk: _parse_or_none(cfg_section, f"{foo}{kk}") for kk in ["fluctuations", "loglogavgslope", "flexibility", "asperity"]}
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


def _spatial_dom(cfg):
    nx = cfg.getint("space npix x")
    ny = cfg.getint("space npix y")
    dx = str2rad(cfg["space fov x"]) / nx
    dy = str2rad(cfg["space fov y"]) / ny
    return ift.RGSpace([nx, ny], [dx, dy])


def _parse_or_none(cfg, key):
    key0 = f"{key} mean"
    key1 = f"{key} stddev"
    if cfg[key0] == "None" and cfg[key1] == "None":
        return None
    if key0 in cfg:
        return (cfg.getfloat(key0), cfg.getfloat(key1))


def _default_sky_domain(pdom=PolarizationSpace("I"), tdom=IRGSpace([np.nan]), fdom=IRGSpace([np.nan]),
                        sdom=ift.RGSpace([1, 1])):
    return pdom, tdom, fdom, sdom
