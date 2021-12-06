# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np

from .constants import str2rad
from .integrated_wiener_process import (IntWProcessInitialConditions,
                                        WienerIntegrations)
from .irg_space import IRGSpace
from .points import PointInserter
from .polarization_matrix_exponential import polarization_matrix_exponential
from .polarization_space import PolarizationSpace
from .simple_operators import DomainChangerAndReshaper, MultiFieldStacker
from .util import assert_sky_domain


def single_frequency_sky(cfg_section):
    sdom = _spatial_dom(cfg_section)
    pdom = PolarizationSpace(cfg_section["polarization"].split(","))
    dom = default_sky_domain(sdom=sdom, pdom=pdom)

    additional = {}

    logsky = []
    for lbl in pdom.labels:
        pol_lbl = f"stokes{lbl.lower()}"
        kwargs1 = {
            "offset_mean": cfg_section.getfloat(f"{pol_lbl} zero mode offset"),
            "offset_std": (cfg_section.getfloat(f"{pol_lbl} zero mode mean"),
                           cfg_section.getfloat(f"{pol_lbl} zero mode stddev"))
        }
        kwargs2 = {kk: _parse_or_none(cfg_section, f"{pol_lbl} space {kk}")
                   for kk in ["fluctuations", "loglogavgslope", "flexibility", "asperity"]}
        op = ift.SimpleCorrelatedField(sdom, **kwargs1, **kwargs2)
        logsky.append(op.ducktape_left(lbl))
        additional[f"logdiffuse stokes{lbl} power spectrum"] = op.power_spectrum
    logsky = reduce(add, logsky)
    mfs = MultiFieldStacker((pdom, sdom), 0, pdom.labels)
    # print("Run test...", end="", flush=True)
    # ift.extra.check_linear_operator(mfs)  # FIXME Move to tests
    # print("done")
    logsky = mfs @ logsky

    mexp = polarization_matrix_exponential(logsky.target)
    # print("Run test...", end="", flush=True)
    # ift.extra.check_operator(mexp, ift.from_random(mexp.domain))  # FIXME Move to tests
    # print("done")
    sky = mexp @ logsky

    additional["logdiffuse"] = mfs.inverse @ logsky
    additional["diffuse"] = mfs.inverse @ sky

    sky = DomainChangerAndReshaper(sky.target, dom) @ sky

    # Point sources
    mode = cfg_section["point sources mode"]
    if mode == "single":
        ppos = []
        s = cfg_section["point sources locations"]
        for xy in s.split(";"):
            x, y = xy.split(",")
            ppos.append((str2rad(x), str2rad(y)))
        alpha = cfg_section.getfloat("point sources alpha")
        q = cfg_section.getfloat("point sources q")

        npoints = len(ppos)
        inserter = PointInserter(dom, ppos)
        ift.extra.check_linear_operator(inserter)

        if pdom.labels_eq("I"):
            points = ift.InverseGammaOperator(inserter.domain, alpha=alpha, q=q/sdom.scalar_dvol)
            points = inserter @ points.ducktape("points")
        elif pdom.labels_eq(["I", "Q", "U"]):
            points_domain = ift.UnstructuredDomain(npoints)
            i = ift.InverseGammaOperator(points_domain, alpha=alpha, q=q/sdom.scalar_dvol).log().ducktape("points I")
            q = ift.NormalTransform(cfg_section["point sources stokesq log mean"], cfg_section["point sources stokesq log stddev"], "points Q", npoints)
            u = ift.NormalTransform(cfg_section["point sources stokesu log mean"], cfg_section["point sources stokesu log stddev"], "points U", npoints)
            conv = DomainChangerAndReshaper(q.target, i.target)
            q = conv.ducktape_left("Q") @ q
            u = conv.ducktape_left("U") @ u
            i = i.ducktape_left("I")
            iqu = MultiFieldStacker((pdom, points_domain), 0, pdom.labels) @ (i + q + u)
            foo = polarization_matrix_exponential(iqu.target) @ iqu
            conv = DomainChangerAndReshaper(foo.target, inserter.domain)
            points = inserter @ conv @ foo
        else:
            raise NotImplementedError(f"single_frequency_sky does not support point sources on {pdom.labels} (yet?)")

        additional["points"] = points
        sky = sky + points
    elif mode == "disable":
        additional["points"] = None
    else:
        raise ValueError(f"In order to disable point source component, set `point sources mode` to `disable`. Got: {mode}")

    multifield_sky = mfs.inverse @ DomainChangerAndReshaper(sky.target, mfs.target) @ sky
    if "U" in multifield_sky.target.keys() and "Q" in multifield_sky.target.keys():
        polarized = (multifield_sky["Q"] ** 2 + multifield_sky["U"] ** 2).sqrt()
        additional["linear polarization"] = polarized
        additional["fractional polarization"] = polarized * multifield_sky["I"].reciprocal()

    assert_sky_domain(sky.target)
    return sky, additional


def multi_frequency_sky(cfg):
    if "point sources mode" in cfg:
        raise NotImplementedError("Point sources are not supported yet.")

    sdom = _spatial_dom(cfg)
    additional = {}

    if cfg["freq mode"] == "cfm":
        fnpix, df = cfg.getfloat("freq npix"), cfg.getfloat("freq pixel size")
        fdom = IRGSpace(cfg.getfloat("freq start") + np.arange(fnpix)*df)
        fdom_rg = ift.RGSpace(fnpix, df)
        logdiffuse, cfm = cfm_from_cfg(cfg, {"freq": fdom_rg, "space": sdom}, "sky diffuse")
        sky = logdiffuse.exp()
        additional["logdiffuse"] = logdiffuse
        fampl, sampl = list(cfm.get_normalized_amplitudes())
        additional["freq normalized power spectrum"] = fampl**2
        additional["space normalized power spectrum"] = sampl**2

        additional["sky"] = sky
        reshape = DomainChangerAndReshaper(sky.target, default_sky_domain(fdom=fdom, sdom=sdom))
        sky = reshape @ sky

    elif cfg["freq mode"] == "integrated wiener process":
        prefix = "sky"

        freq = np.sort(np.array(list(map(float, cfg["frequencies"].split(",")))))
        fdom = IRGSpace(freq)
        log_fdom = IRGSpace(np.log(freq / freq.mean()))

        # Base sky brightness distribution for mean frequency \nu_0
        i_0, i_0_cfm = cfm_from_cfg(cfg, {"space i0": sdom}, "space i0")

        # simple spectral index that is frequency independent
        alpha, alpha_cfm = cfm_from_cfg(cfg, {"space alpha": sdom}, "space alpha")

        flexibility = _parse_or_none(cfg, "freq flexibility")
        if flexibility is None:
            raise RuntimeError("freq flexibility cannot be None")
        asperity = _parse_or_none(cfg, "freq asperity")

        # IDEA Try to use individual power spectra
        n_freq_xi_fields = 2 * (log_fdom.size - 1)
        cfm = ift.CorrelatedFieldMaker(f"{prefix}freq_xi", total_N=n_freq_xi_fields)
        cfm.set_amplitude_total_offset(0.0, None)
        # FIXME Support fixed fluctuations
        cfm.add_fluctuations(sdom, (1, 1e-6), (1.2, 0.4), (0.2, 0.2), (-2, 0.5), dofdex=n_freq_xi_fields * [0])
        freq_xi = cfm.finalize(0)

        # Integrate over excitation fields
        intop = WienerIntegrations(log_fdom, sdom)
        freq_xi = DomainChangerAndReshaper(freq_xi.target, intop.domain) @ freq_xi
        broadcast = ift.ContractionOperator(intop.domain[0], None).adjoint
        broadcast_full = ift.ContractionOperator(intop.domain, 1).adjoint
        vol = log_fdom.distances

        flex = ift.LognormalTransform(*flexibility, prefix + "flexibility", 0)

        dom = intop.domain[0]
        vflex = np.empty(dom.shape)
        vflex[0] = vflex[1] = np.sqrt(vol)
        sig_flex = ift.makeOp(ift.makeField(dom, vflex)) @ broadcast @ flex
        sig_flex = broadcast_full @ sig_flex
        shift = np.ones(dom.shape)
        shift[0] = vol * vol / 12.0
        if asperity is None:
            shift = ift.DiagonalOperator(ift.makeField(dom, shift).sqrt(), intop.domain, 0)
            increments = shift @ (freq_xi * sig_flex)
        else:
            asp = ift.LognormalTransform(*asperity, prefix + "asperity", 0)
            vasp = np.empty(dom.shape)
            vasp[0] = 1
            vasp[1] = 0
            vasp = ift.DiagonalOperator(ift.makeField(dom, vasp), domain=broadcast.target, spaces=0)
            sig_asp = broadcast_full @ vasp @ broadcast @ asp
            shift = ift.makeField(intop.domain, np.broadcast_to(shift[..., None, None], intop.domain.shape))
            increments = freq_xi * sig_flex * (ift.Adder(shift) @ sig_asp).ptw("sqrt")
        logsky = IntWProcessInitialConditions(i_0, alpha, intop @ increments)

        sky = logsky.exp()
        additional["sky"] = sky
        additional["logsky"] = logsky
        additional["i0"] = i_0
        additional["alpha"] = alpha

        reshape = DomainChangerAndReshaper(sky.target, default_sky_domain(fdom=fdom, sdom=sdom))
        sky = reshape @ sky

    else:
        raise RuntimeError

    assert_sky_domain(sky.target)
    return sky, additional


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


def default_sky_domain(pdom=PolarizationSpace("I"), tdom=IRGSpace([np.nan]), fdom=IRGSpace([np.nan]),
                       sdom=ift.RGSpace([1, 1])):
    from .util import my_assert_isinstance

    for dd in [pdom, tdom, fdom, sdom]:
        my_assert_isinstance(dd, ift.Domain)
    return ift.DomainTuple.make((pdom, tdom, fdom, sdom))
