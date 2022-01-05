# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np

from .simple_operators import MultiFieldStacker
from .sky_model import cfm_from_cfg
from .util import assert_sky_domain


def weighting_model(cfg, obs, sky_domain):
    """Assumes independent weighting for every imaging band and for every polarization"""
    assert_sky_domain((sky_domain))
    n_imaging_bands = sky_domain[2].size
    n_data_pol = obs.npol

    if cfg.getboolean("enable"):
        if cfg["model"] == "cfm":
            import ducc0

            npix = cfg.getint("npix")
            fac = cfg.getfloat("zeropadding factor")
            npix_padded = ducc0.fft.good_size(int(np.round(npix*fac)))

            xs = np.log(obs.effective_uvwlen().val)
            minlen, maxlen = np.min(xs), np.max(xs)
            xs -= minlen

            dom = ift.RGSpace(npix_padded, fac * maxlen / npix)

            cfm = cfm_from_cfg(cfg, {"": dom}, "invcov", total_N=n_imaging_bands*n_data_pol)
            log_weights = cfm.finalize(0)
            keys = [_polfreq_key(pp, ii) for pp in range(n_data_pol) for ii in range(n_imaging_bands)]
            mfs = MultiFieldStacker(log_weights.target, 0, keys)
            pspecs = MultiFieldStacker(cfm.power_spectrum.target, 0, keys).inverse @ cfm.power_spectrum
            #ift.extra.check_linear_operator(mfs)
            op = []
            for pp in range(n_data_pol):
                for ii in range(n_imaging_bands):
                    foo = ift.LinearInterpolator(dom, xs[0, :, ii][None])
                    key = _polfreq_key(pp, ii)
                    op.append(foo.ducktape(key).ducktape_left(key))
            linear_interpolation = reduce(add, op)
            restructure = _CustomRestructure(linear_interpolation.target, obs.vis.domain)
            ift.extra.check_linear_operator(restructure)
            log_weights = mfs.inverse @ log_weights
            op = ift.makeOp(obs.weight) @ (restructure @ linear_interpolation @ log_weights).scale(-2).exp()
            additional = {
                "weights_power_spectrum": pspecs,
                "log_sigma_correction": log_weights,
                "sigma_correction": log_weights.exp()
            }
            return op, additional
        else:
            raise NotImplementedError
    return None, {}


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
