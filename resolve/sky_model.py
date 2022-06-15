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
# Copyright(C) 2021 Max-Planck-Society
# Copyright(C) 2022 Philipp Arras
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
from .polarization_matrix_exponential import \
    polarization_matrix_exponential_mf2f
from .polarization_space import PolarizationSpace
from .simple_operators import MultiFieldStacker
from .util import assert_sky_domain


def sky_model_diffuse(cfg, observations=[], nthreads=1):
    sdom = _spatial_dom(cfg)
    pdom = PolarizationSpace(cfg["polarization"].split(","))

    additional = {}

    logsky = {}
    for lbl in pdom.labels:
        pol_lbl = f"{lbl.upper()}"
        if cfg["freq mode"] == "single":
            op, aa = _single_freq_logsky(cfg, pol_lbl)
        elif cfg["freq mode"] == "cfm":
            op, aa = _multi_freq_logsky_cfm(cfg, sdom, pol_lbl)
        elif cfg["freq mode"] == "iwp":
            freq = _get_frequencies(cfg, observations)
            op, aa = _multi_freq_logsky_integrated_wiener_process(cfg, sdom, pol_lbl, freq)
        else:
            raise RuntimeError
        logsky[lbl] = op
        additional = {**additional, **aa}
    if cfg["freq mode"] == "single":
        tgt = default_sky_domain(pdom=pdom, sdom=sdom)
    else:
        fdom = op.target[0]
        tgt = default_sky_domain(pdom=pdom, fdom=fdom, sdom=sdom)

    logsky = reduce(add, (oo.ducktape_left(lbl) for lbl, oo in logsky.items()))
    mexp = polarization_matrix_exponential_mf2f(logsky.target, nthreads=nthreads)
    sky = mexp @ logsky

    sky = sky.ducktape_left(tgt)
    assert_sky_domain(sky.target)

    return sky, additional


def sky_model_points(cfg, observations=[], nthreads=1):
    sdom = _spatial_dom(cfg)
    pdom = PolarizationSpace(cfg["polarization"].split(","))

    additional = {}
    # Point sources
    if cfg["point sources mode"] == "single":
        ppos = []
        s = cfg["point sources locations"]
        for xy in s.split(","):
            x, y = xy.split("$")
            ppos.append((str2rad(x), str2rad(y)))
        alpha = cfg.getfloat("point sources alpha")
        q = cfg.getfloat("point sources q")

        inserter = PointInserter(default_sky_domain(pdom=pdom, sdom=sdom), ppos)

        if pdom.labels_eq("I"):
            points = ift.InverseGammaOperator(inserter.domain, alpha=alpha, q=q/sdom.scalar_dvol)
            points = points.ducktape("points")
        elif pdom.labels_eq(["I", "Q", "U"]) or pdom.labels_eq(["I", "Q", "U", "V"]):
            points_domain = inserter.domain[-1]
            npoints = points_domain.size
            i = ift.InverseGammaOperator(points_domain, alpha=alpha, q=q/sdom.scalar_dvol).log().ducktape("points I")
            q = ift.NormalTransform(cfg["point sources stokesq log mean"], cfg["point sources stokesq log stddev"], "points Q", npoints)
            u = ift.NormalTransform(cfg["point sources stokesu log mean"], cfg["point sources stokesu log stddev"], "points U", npoints)
            i = i.ducktape_left("I")
            q = q.ducktape_left("Q")
            u = u.ducktape_left("U")
            polsum = i + q + u
            if pdom.labels_eq(["I", "Q", "U", "V"]):
                v = ift.NormalTransform(cfg["point sources stokesv log mean"], cfg["point sources stokesv log stddev"], "points V", npoints)
                v = v.ducktape_left("V")
                polsum = polsum + v
            points = polarization_matrix_exponential_mf2f(polsum.target, nthreads=nthreads) @ polsum
            points = points.ducktape_left(inserter.domain)
        else:
            raise NotImplementedError(f"single_frequency_sky does not support point sources on {pdom.labels} (yet?)")

        additional["point_list"] = points
        sky = inserter @ points
    elif cfg["point sources mode"] == "single-iwp":
        if not pdom.labels_eq("I"):
            raise NotImplementedError
        ppos = []
        s = cfg["point sources locations"]
        for xy in s.split(","):
            x, y = xy.split("$")
            ppos.append((str2rad(x), str2rad(y)))
        alpha = cfg.getfloat("point sources alpha")
        q = cfg.getfloat("point sources q")

        inserter = PointInserter(sky.target, ppos)
        udom = inserter.domain[-1]

        p_i0 = ift.InverseGammaOperator(udom, alpha=alpha, q=q/sdom.scalar_dvol)
        p_i0 = p_i0.ducktape("points")

        p_alpha = _parse_or_none(cfg, "point sources alpha")
        p_alpha = ift.NormalTransform(*p_alpha, "points alpha", udom.shape)
        p_alpha = p_alpha.ducktape_left(udom)

        p_flex = _parse_or_none(cfg, "point sources flexibility")
        p_flex = ift.LognormalTransform(*p_flex, "points flexibility", 0)

        p_asp = _parse_or_none(cfg, "point sources asperity")
        if p_asp is not None:
            p_asp = ift.LognormalTransform(*p_asp, "points asperity", 0)

        freq = _get_frequencies(cfg, observations)
        log_fdom = IRGSpace(np.sort(np.log(freq)))
        nfreq = len(freq)
        npoints = udom.size
        p_xi = ift.ScalingOperator(ift.UnstructuredDomain(2*npoints*(nfreq - 1)), 1.).ducktape("points xi")

        points = _integrated_wiener_process(p_i0, p_alpha, log_fdom, p_flex, p_asp, p_xi)
        points = points.ducktape_left(inserter.domain)

        additional["point_list"] = points
        sky = inserter @ points
    else:
        return None, {}

    assert_sky_domain(sky.target)
    return sky, additional


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
    freq0 = cfg.getfloat("freq start")
    if fnpix is None:
        raise ValueError("Please set a value for `freq npix`")
    if df is None:
        raise ValueError("Please set a value for `freq pixel size`")
    if freq0 is None:
        raise ValueError("Please set a value for `freq start`")
    fdom = IRGSpace(freq0 + np.arange(fnpix)*df)
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

    flexibility = _parse_or_none(cfg, prefix + " wp flexibility")
    if flexibility is None:
        raise RuntimeError("freq flexibility cannot be None")
    flexibility = ift.LognormalTransform(*flexibility, prefix + " wp flexibility", 0)

    asperity = _parse_or_none(cfg, prefix + " wp asperity")
    asperity = ift.LognormalTransform(*asperity, prefix + " wp asperity", 0)
    additional = {
        f"stokes{pol_label} i0": i0,
        f"stokes{pol_label} alpha": alpha,
    }
    iwp = _integrated_wiener_process(i0, alpha, log_fdom,
                                     flexibility, asperity, freq_xi)
    iwp = iwp.ducktape_left((fdom, sdom))
    return iwp, additional


def _integrated_wiener_process(i0, alpha, irg_space, flexibility, asperity, freq_xi):
    assert i0.target == alpha.target
    assert len(i0.target) == 1
    # Integrate over excitation fields
    intop = WienerIntegrations(irg_space, i0.target[0])
    freq_xi = freq_xi.ducktape_left(intop.domain)
    broadcast = ift.ContractionOperator(intop.domain[0], None).adjoint
    broadcast_full = ift.ContractionOperator(intop.domain, 1).adjoint
    vol = irg_space.distances

    dom = intop.domain[0]
    vflex = np.empty(dom.shape)
    vflex[0] = vflex[1] = np.sqrt(vol)
    sig_flex = ift.makeOp(ift.makeField(dom, vflex)) @ broadcast @ flexibility
    sig_flex = broadcast_full @ sig_flex
    shift = np.ones(dom.shape)
    shift[0] = vol * vol / 12.0
    if asperity is None:
        shift = ift.DiagonalOperator(ift.makeField(dom, shift).sqrt(), intop.domain, 0)
        increments = shift @ (freq_xi * sig_flex)
    else:
        vasp = np.empty(dom.shape)
        vasp[0] = 1
        vasp[1] = 0
        vasp = ift.DiagonalOperator(ift.makeField(dom, vasp), domain=broadcast.target, spaces=0)
        sig_asp = broadcast_full @ vasp @ broadcast @ asperity
        shift = ift.makeField(intop.domain, np.broadcast_to(shift[..., None, None], intop.domain.shape))
        increments = freq_xi * sig_flex * (ift.Adder(shift) @ sig_asp).ptw("sqrt")

    return IntWProcessInitialConditions(i0, alpha, intop @ increments)


def cfm_from_cfg(cfg, domain_dct, prefix, total_N=0, dofdex=None, override={}, domain_prefix=None):
    assert len(prefix) > 0
    product_spectrum = len(domain_dct) > 1
    cfm = ift.CorrelatedFieldMaker(prefix if domain_prefix is None else domain_prefix, total_N=total_N)
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


def default_sky_domain(pdom=None, tdom=None, fdom=None, sdom=None):
    from .util import my_assert_isinstance

    if pdom is None:
        pdom = PolarizationSpace("I")
    if tdom is None:
        tdom = IRGSpace([0.])
    if fdom is None:
        fdom = IRGSpace([1.])
    if sdom is None:
        sdom = ift.RGSpace([1, 1])

    return ift.DomainTuple.make((pdom, tdom, fdom, sdom))


def _get_frequencies(cfg, observations):
    if cfg["frequencies"] == "data":
        freq = np.array([oo.freq for oo in observations]).flatten()
    else:
        freq = map(float, cfg["frequencies"].split(","))
    return np.sort(np.array(list(freq)))
