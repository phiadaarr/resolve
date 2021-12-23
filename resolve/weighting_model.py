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
    assert_sky_domain((sky_domain))
    n_imaging_bands = sky_domain[2].size

    if cfg.getboolean("enable"):
        if obs.npol > 1:
            raise NotImplementedError("Weighting not supported for multiple polarizations yet")
        if cfg["model"] == "cfm":
            import ducc0

            npix = cfg.getint("npix")
            fac = cfg.getfloat("zeropadding factor")
            npix_padded = ducc0.fft.good_size(int(np.round(npix*fac)))

            xs = np.log(obs.effective_uvwlen().val)
            minlen, maxlen = np.min(xs), np.max(xs)
            xs -= minlen

            dom = ift.RGSpace(npix_padded, fac * maxlen / npix)

            cfm = cfm_from_cfg(cfg, {"": dom}, "invcov", total_N=n_imaging_bands)
            log_weights = cfm.finalize(0)
            mfs = MultiFieldStacker(log_weights.target, 0, [str(ii) for ii in range(n_imaging_bands)])
            mfs1 = MultiFieldStacker(obs.vis.domain[1:], 1, [str(ii) for ii in range(n_imaging_bands)])
            #ift.extra.check_linear_operator(mfs)
            #ift.extra.check_linear_operator(mfs1)
            op = []
            for ii in range(n_imaging_bands):
                foo = ift.LinearInterpolator(dom, xs[0, :, ii][None])
                op.append(foo.ducktape(str(ii)).ducktape_left(str(ii)))
            log_weights = (mfs1 @ reduce(add, op) @ mfs.inverse @ log_weights).ducktape_left(obs.vis.domain)
            op = ift.makeOp(obs.weight) @ log_weights.scale(-2).exp()
            additional = {
                # "weights power spectrum": cfm.power_spectrum
                #operators["log_sigma_correction"] = log_weights
                #operators["sigma_correction"] = log_weights.exp()
            }
            return op, additional
        else:
            raise NotImplementedError
    return None, {}
