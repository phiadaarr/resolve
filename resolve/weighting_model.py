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
from .dtype_converter import DtypeConverter


def weighting_model(cfg, obs, sky_domain):
    op, additional = log_weighting_model
    return op.exp(), additional


def log_weighting_model(cfg, obs, sky_domain):
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
            tmpop = (restructure @ linear_interpolation @ log_weights).scale(-2)
            if oo.is_single_precision():
                tmpop = DtypeConverter(tmpop.target, np.float64, np.float32) @ tmpop
            tmpop = ift.Adder(_log_if_not_zero(oo.weight)) @ tmpop
            op.append(tmpop)
        return op, additional
    if cfg["model"] == "independent gamma":
        mean = cfg.getfloat("mean")
        var = cfg.getfloat("var")
        alpha = cfg.getfloat("alpha")
        theta = cfg.getfloat("theta")
        ops = []
        for iobs, oo in enumerate(obs):
            op = ift.GammaOperator(oo.vis.domain, mean=mean, var=var, alpha=alpha, theta=theta)
            if oo.is_single_precision():
                op = DtypeConverter(op.target, np.float64, np.float32) @ op
            op = ift.makeOp(oo.weight) @ op
            op = op.log()  # FIXME Simplify this for better performance
            op = op.ducktape(f"Observation {iobs}, invcov")
            ops.append(op)
        return ops, {}
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


def _log_if_not_zero(fld):
    res = fld.log().val_rw()
    res[fld.val == 0.] = 0.
    return ift.makeField(fld.domain, res)
