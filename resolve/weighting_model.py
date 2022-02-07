# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import ducc0
import nifty8 as ift
import numpy as np

from .data.observation import Observation
from .simple_operators import MultiFieldStacker
from .sky_model import cfm_from_cfg
from .util import _obj2list, assert_sky_domain


def weighting_model(cfg, obs, sky_domain):
    """Assumes independent weighting for every imaging band and for every polarization"""
    assert_sky_domain(sky_domain)

    if not cfg.getboolean("enable"):
        return None, {}

    obs = _obj2list(obs, Observation)

    if cfg["model"] == "cfm":
        npix = cfg.getint("npix")
        fac = cfg.getfloat("zeropadding factor")
        npix_padded = ducc0.fft.good_size(int(np.round(npix*fac)))

        op, additional = [], {}
        for iobs, oo in enumerate(obs):
            xs = oo.effective_uvwlen().val_rw()
            minlen = np.min(xs)
            xs -= minlen
            maxlen = np.max(xs)
            dom = ift.RGSpace(npix_padded, maxlen / npix)

            cfm = cfm_from_cfg(cfg, {"": dom}, "invcov", total_N=oo.nfreq*oo.npol,
                               domain_prefix=f"Observation {iobs}, invcov")
            log_weights = cfm.finalize(0)

            keys = [_polfreq_key(pp, ii) for pp in range(oo.npol)
                                         for ii in range(oo.nfreq)]
            mfs = MultiFieldStacker(log_weights.target, 0, keys)
            log_weights = mfs.inverse @ log_weights
            pspecs = MultiFieldStacker(cfm.power_spectrum.target, 0, keys).inverse @ cfm.power_spectrum
            additional[f"observation {iobs}: weights_power_spectrum"] = pspecs
            additional[f"observation {iobs}: log_sigma_correction"] = log_weights
            additional[f"observation {iobs}: sigma_correction"] = log_weights.exp()
            tmpop = []
            for pp in range(oo.npol):
                for ii in range(oo.nfreq):
                    assert dom.total_volume >= xs[0, :, ii].max()
                    assert 0 <= xs[0, :, ii].min()
                    foo = ift.LinearInterpolator(dom, xs[0, :, ii][None])
                    key = _polfreq_key(pp, ii)
                    tmpop.append(foo.ducktape(key).ducktape_left(key))
            linear_interpolation = reduce(add, tmpop)
            restructure = _CustomRestructure(linear_interpolation.target, oo.vis.domain)
            ift.extra.check_linear_operator(restructure)
            tmpop = ift.makeOp(oo.weight) @ (restructure @ linear_interpolation @ log_weights).scale(-2).exp()
            op.append(tmpop)
        return op, additional
    raise NotImplementedError


def _polfreq_key(stokes_label, freq):
    return f"Stokes {stokes_label}, freqband {freq}"


class _CustomRestructure(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.MultiDomain.make(domain)
        self._target = ift.DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = np.empty(self.target.shape)
            for pp in range(self.target.shape[0]):
                for ii in range(self.target.shape[2]):
                    res[pp, :, ii] = x[_polfreq_key(pp, ii)]
        else:
            res = {}
            for pp in range(self.target.shape[0]):
                for ii in range(self.target.shape[2]):
                    res[_polfreq_key(pp, ii)] = x[pp, :, ii]
        return ift.makeField(self._tgt(mode), res)
