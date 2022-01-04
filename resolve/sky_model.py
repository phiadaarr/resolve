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
from .simple_operators import MultiFieldStacker
from .util import assert_sky_domain


def sky_model(cfg, observations=[]):
    sdom = _spatial_dom(cfg)
    pdom = PolarizationSpace(cfg["polarization"].split(","))

    additional = {}
    domains = {}

    logsky = []
    for lbl in pdom.labels:
        pol_lbl = f"{lbl.upper()}"
        if cfg["freq mode"] == "single":
            op, aa = _single_freq_logsky(cfg, pol_lbl)
        elif cfg["freq mode"] == "cfm":
            op, aa = _multi_freq_logsky_cfm(cfg, sdom, pol_lbl)
        elif cfg["freq mode"] == "iwp":
            if cfg["frequencies"] == "data":
                freq = np.array([oo.freq for oo in observations]).flatten()
            else:
                freq = map(float, cfg["frequencies"].split(","))
            freq = np.sort(np.array(list(freq)))
            op, aa = _multi_freq_logsky_integrated_wiener_process(cfg, sdom, pol_lbl, freq)
        else:
            raise RuntimeError
        logsky.append(op.ducktape_left(pol_lbl))
        additional = {**additional, **aa}
    if cfg["freq mode"] == "single":
        tgt = default_sky_domain(pdom=pdom, sdom=sdom)
        mfs = MultiFieldStacker((pdom, sdom), 0, pdom.labels)
    else:
        fdom = op.target[0]
        tgt = default_sky_domain(pdom=pdom, fdom=fdom, sdom=sdom)
        mfs = MultiFieldStacker((pdom, fdom, sdom), 0, pdom.labels)
    logsky = reduce(add, logsky)
    # additional["logdiffuse"] = logsky

    mfs1 = MultiFieldStacker(tgt, 0, pdom.labels)
    #ift.extra.check_linear_operator(mfs)
    #ift.extra.check_linear_operator(mfs1)
    mexp = polarization_matrix_exponential(tgt)

    sky = mexp @ (mfs @ logsky).ducktape_left(tgt)
    domains["diffuse"] = sky.domain
    # additional["diffuse"] = mfs1.inverse @ sky

    # Point sources
    if cfg["point sources mode"] == "single":
        ppos = []
        s = cfg["point sources locations"]
        for xy in s.split(","):
            x, y = xy.split("$")
            ppos.append((str2rad(x), str2rad(y)))
        alpha = cfg.getfloat("point sources alpha")
        q = cfg.getfloat("point sources q")

        inserter = PointInserter(sky.target, ppos)

        if pdom.labels_eq("I"):
            points = ift.InverseGammaOperator(inserter.domain, alpha=alpha, q=q/sdom.scalar_dvol)
            points = inserter @ points.ducktape("points")
        elif pdom.labels_eq(["I", "Q", "U"]):
            points_domain = inserter.domain[-1]
            npoints = points_domain.size
            i = ift.InverseGammaOperator(points_domain, alpha=alpha, q=q/sdom.scalar_dvol).log().ducktape("points I")
            q = ift.NormalTransform(cfg["point sources stokesq log mean"], cfg["point sources stokesq log stddev"], "points Q", npoints)
            u = ift.NormalTransform(cfg["point sources stokesu log mean"], cfg["point sources stokesu log stddev"], "points U", npoints)
            i = i.ducktape_left("I")
            q = q.ducktape_left("Q")
            u = u.ducktape_left("U")
            iqu = MultiFieldStacker((pdom, points_domain), 0, pdom.labels) @ (i + q + u)
            foo = polarization_matrix_exponential(iqu.target) @ iqu
            points = inserter.ducktape(foo.target) @ foo
        else:
            raise NotImplementedError(f"single_frequency_sky does not support point sources on {pdom.labels} (yet?)")

        additional["points"] = points
        domains["points"] = points.domain
        sky = sky + points

    if not sky.target[0].labels_eq("I"):
        multifield_sky = mfs.inverse.ducktape(sky.target) @ sky
        additional["sky"] = multifield_sky
        if "U" in multifield_sky.target.keys() and "Q" in multifield_sky.target.keys():
            polarized = (multifield_sky["Q"] ** 2 + multifield_sky["U"] ** 2).sqrt()
            additional["linear polarization"] = polarized
            additional["fractional polarization"] = polarized * multifield_sky["I"].reciprocal()
    else:
        if sky.target[1].size == 1 and sky.target[2].size > 1:
            additional["mf sky"] = MultiFieldStacker(sky.target[2:], 0,
                                                     [f"{cc*1e-9:.6} GHz" for cc in sky.target[2].coordinates]).inverse \
                                       @ sky.ducktape_left(sky.target[2:])
            additional["mf logsky"] = additional["mf sky"].log()

    assert_sky_domain(sky.target)
    domains["sky"] = sky.domain
    return sky, additional, domains


def _single_freq_logsky(cfg, pol_label):
    sdom = _spatial_dom(cfg)
    cfm = cfm_from_cfg(cfg, {"space i0": sdom}, f"stokes{pol_label} diffuse")
    op = cfm.finalize(0)
    additional = {
        f"logdiffuse stokes{pol_label} power spectrum": cfm.power_spectrum,
        f"logdiffuse stokes{pol_label}": op,
    }
    return op, additional


def _multi_freq_logsky_cfm(cfg, sdom, pol_label):
    fnpix, df = cfg.getfloat("freq npix"), cfg.getfloat("freq pixel size")
    fdom = IRGSpace(cfg.getfloat("freq start") + np.arange(fnpix)*df)
    fdom_rg = ift.RGSpace(fnpix, df)

    cfm = cfm_from_cfg(cfg, {"freq": fdom_rg, "space": sdom}, f"stokes{pol_label} diffuse")
    op = cfm.finalize(0)

    fampl, sampl = list(cfm.get_normalized_amplitudes())
    additional = {
        f"logdiffuse stokes{pol_label}": op,
        f"freq normalized power spectrum stokes{pol_label}": fampl**2,
        f"space normalized power spectrum stokes{pol_label}": sampl**2
    }
    return op.ducktape_left((fdom, sdom)), additional


def _multi_freq_logsky_integrated_wiener_process(cfg, sdom, pol_label, freq):
    assert len(freq) > 0

    fdom = IRGSpace(freq)

    prefix = f"stokes{pol_label} diffuse"
    i0_cfm = cfm_from_cfg(cfg, {"": sdom}, prefix + " space i0")
    alpha_cfm = cfm_from_cfg(cfg, {"": sdom}, prefix + " space alpha")
    i0 = i0_cfm.finalize(0)
    alpha = alpha_cfm.finalize(0)

    log_fdom = IRGSpace(np.log(freq / freq.mean()))
    n_freq_xi_fields = 2 * (log_fdom.size - 1)
    override = {
        f"stokes{pol_label} diffuse wp increments zero mode offset": 0.,
        f"stokes{pol_label} diffuse wp increments zero mode": (None, None),
        # FIXME NIFTy cfm: support fixed fluctuations
        f"stokes{pol_label} diffuse wp increments fluctuations": (1, 1e-6),
    }
    # IDEA Try to use individual power spectra
    cfm = cfm_from_cfg(cfg, {"": sdom}, prefix + " wp increments",
                       total_N=n_freq_xi_fields, dofdex=n_freq_xi_fields * [0],
                       override=override)
    freq_xi = cfm.finalize(0)

    # Integrate over excitation fields
    intop = WienerIntegrations(log_fdom, sdom)
    freq_xi = freq_xi.ducktape_left(intop.domain)
    broadcast = ift.ContractionOperator(intop.domain[0], None).adjoint
    broadcast_full = ift.ContractionOperator(intop.domain, 1).adjoint
    vol = log_fdom.distances

    flexibility = _parse_or_none(cfg, prefix + " wp flexibility")
    if flexibility is None:
        raise RuntimeError("freq flexibility cannot be None")
    flex = ift.LognormalTransform(*flexibility, prefix + " wp flexibility", 0)
    dom = intop.domain[0]
    vflex = np.empty(dom.shape)
    vflex[0] = vflex[1] = np.sqrt(vol)
    sig_flex = ift.makeOp(ift.makeField(dom, vflex)) @ broadcast @ flex
    sig_flex = broadcast_full @ sig_flex
    shift = np.ones(dom.shape)
    shift[0] = vol * vol / 12.0
    asperity = _parse_or_none(cfg, prefix + " wp asperity")
    if asperity is None:
        shift = ift.DiagonalOperator(ift.makeField(dom, shift).sqrt(), intop.domain, 0)
        increments = shift @ (freq_xi * sig_flex)
    else:
        asp = ift.LognormalTransform(*asperity, prefix + " wp asperity", 0)
        vasp = np.empty(dom.shape)
        vasp[0] = 1
        vasp[1] = 0
        vasp = ift.DiagonalOperator(ift.makeField(dom, vasp), domain=broadcast.target, spaces=0)
        sig_asp = broadcast_full @ vasp @ broadcast @ asp
        shift = ift.makeField(intop.domain, np.broadcast_to(shift[..., None, None], intop.domain.shape))
        increments = freq_xi * sig_flex * (ift.Adder(shift) @ sig_asp).ptw("sqrt")
    op = IntWProcessInitialConditions(i0, alpha, intop @ increments)

    additional = {
        f"stokes{pol_label} i0": i0,
        f"stokes{pol_label} alpha": alpha,
    }
    op = op.ducktape_left((fdom, sdom))
    return op, additional


def cfm_from_cfg(cfg, domain_dct, prefix, total_N=0, dofdex=None, override={}):
    assert len(prefix) > 0
    product_spectrum = len(domain_dct) > 1
    cfm = ift.CorrelatedFieldMaker(f"{prefix}", total_N=total_N)
    for key_prefix, dom in domain_dct.items():
        ll = _append_to_nonempty_string(key_prefix, " ")
        kwargs = {kk: _parse_or_none(cfg, f"{prefix} {ll}{kk}", override)
                  for kk in ["fluctuations", "loglogavgslope", "flexibility", "asperity"]}
        cfm.add_fluctuations(dom, **kwargs, prefix=key_prefix, dofdex=dofdex)
    foo = str(prefix)
    if not product_spectrum and len(key_prefix) != 0:
        foo += f" {key_prefix}"
    kwargs = {
        "offset_mean": _parse_or_none(cfg, f"{foo} zero mode offset", override=override, single_value=True),
        "offset_std": _parse_or_none(cfg, f"{foo} zero mode", override=override)
    }
    cfm.set_amplitude_total_offset(**kwargs)
    return cfm


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


def _parse_or_none(cfg, key, override={}, single_value=False):
    if single_value:
        if key in override:
            return override[key]
        if cfg[key] == "None":
            return None
        return cfg.getfloat(key)
    key0 = f"{key} mean"
    key1 = f"{key} stddev"
    if key in override:
        a, b = override[key]
        if a is None and b is None:
            return None
        return a, b
    if cfg[key0] == "None" and cfg[key1] == "None":
        return None
    if key0 in cfg:
        return (cfg.getfloat(key0), cfg.getfloat(key1))


def default_sky_domain(pdom=PolarizationSpace("I"), tdom=IRGSpace([0.]),
                       fdom=IRGSpace([1.]), sdom=ift.RGSpace([1, 1])):
    from .util import my_assert_isinstance

    for dd in [pdom, tdom, fdom, sdom]:
        my_assert_isinstance(dd, ift.Domain)
    return ift.DomainTuple.make((pdom, tdom, fdom, sdom))
