# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np
from ducc0.wgridder.experimental import dirty2vis, vis2dirty

import nifty7 as ift

from .global_config import epsilon, nthreads, wgridding
from .multi_frequency.irg_space import IRGSpace
from .observation import Observation
from .util import my_assert, my_assert_isinstance, my_asserteq


def StokesIResponse(observation, domain):
    my_assert_isinstance(observation, Observation)
    domain = ift.DomainTuple.make(domain)
    my_asserteq(len(domain), 1)
    my_assert_isinstance(domain[0], ift.RGSpace)
    npol = observation.npol
    my_assert(npol in [1, 2])
    sp = observation.vis.dtype == np.complex64
    mask = observation.mask
    sr0 = SingleResponse(domain, observation.uvw, observation.freq, mask[0], sp)
    # FIXME Get rid of npol==2 support here
    if npol == 1 or (npol == 2 and np.all(mask[0] == mask[1])):
        contr = ift.ContractionOperator(observation.vis.domain, 0)
        return contr.adjoint @ sr0
    elif npol == 2:
        sr1 = SingleResponse(domain, observation.uvw, observation.freq, mask[1], sp)
        return ResponseDistributor(sr0, sr1)
    raise RuntimeError


class FullPolResponse(ift.LinearOperator):
    def __init__(self, observation, domain):
        my_assert_isinstance(observation, Observation)
        domain = ift.MultiDomain.make(domain)
        self._domain = domain
        self._target = observation.vis.domain
        # TODO Add support for Stokes V
        my_asserteq(set(domain.keys()), set(["I", "Q", "U"]))
        my_asserteq(len(domain["I"]), 1)
        my_asserteq(len(domain["I"].shape), 2)
        my_asserteq(dd for dd in domain)
        npol = observation.npol
        my_asserteq(npol, 4)
        sp = observation.vis.dtype == np.complex64
        domain = domain["I"]
        mask = np.all(observation.mask, axis=0)
        self._sr = SingleResponse(domain, observation.uvw, observation.freq, mask, sp)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.empty(self._target.shape, dtype=np.complex64)
            res[0] = res[3] = self._sr(x["I"]).val
            Q = self._sr(x["Q"]).val
            assert Q.dtype == np.complex64
            U = self._sr(x["U"]).val
            res[1] = U-1j*Q
            res[2] = -U-1j*Q
        else:
            op = lambda inp: self._sr.adjoint(ift.makeField(self._sr.target, inp))
            x = x.val
            res = {}
            res["I"] = op(x[0] + x[3]).val
            res["Q"] = op(1j*(x[1] + x[2])).val
            res["U"] = op(x[1] - x[2]).val
        return ift.makeField(self._tgt(mode), res)


class MfResponse(ift.LinearOperator):
    """Multi-frequency response

    This class represents the linear operator that maps the discretized
    brightness distribution into visibilities. It supports mapping a single
    frequency in its domain to multiple channels in its target with a
    nearest-neighbour interpolation. This may be useful for contiuum imaging.

    Parameters
    ----------
    observation : Observation
        Instance of the :class:`Observation` that represents the measured data.
    frequency_domain : IRGSpace
        Contains the :class:`IRGSpace` for the frequencies.
    position_domain : nifty7.RGSpace
        Contains the the :class:`nifty7.RGSpace` for the positions.
    """

    def __init__(self, observation, frequency_domain, position_domain):
        my_assert_isinstance(observation, Observation)
        # FIXME Add polarization support
        my_asserteq(observation.npol, 1)
        my_assert_isinstance(frequency_domain, IRGSpace)
        my_assert_isinstance(position_domain, ift.RGSpace)

        domain_tuple = (frequency_domain, position_domain)
        self._domain = ift.DomainTuple.make(domain_tuple)
        self._target = observation.vis.domain
        self._capability = self.TIMES | self.ADJOINT_TIMES

        data_freq = observation.freq
        my_assert(np.all(np.diff(data_freq) > 0))
        sky_freq = np.array(frequency_domain.coordinates)
        band_indices = [np.argmin(np.abs(ff - sky_freq)) for ff in data_freq]
        # Make sure that no index is wasted
        my_asserteq(len(set(band_indices)), frequency_domain.size)
        self._r = []
        sp = observation.vis.dtype == np.complex64
        mask = observation.mask
        for band_index in np.unique(band_indices):
            sel = band_indices == band_index
            assert mask.shape[0] == 1
            r = SingleResponse(
                position_domain, observation.uvw, observation.freq[sel], mask[0, :, sel].T, sp
            )
            self._r.append((band_index, sel, r))
        # Double check that all channels are written to
        check = np.zeros(len(data_freq))
        for _, sel, _ in self._r:
            check[sel] += 1
        my_assert(np.all(check == 1))

    def apply(self, x, mode):
        self._check_input(x, mode)
        res = None
        x = x.val
        if mode == self.TIMES:
            for band_index, sel, rr in self._r:
                foo = rr(ift.makeField(rr.domain, x[band_index]))
                if res is None:
                    res = np.empty(self._tgt(mode).shape, foo.dtype)
                res[0][..., sel] = foo.val
        else:
            empty = np.zeros(self._domain.shape[0], bool)
            res = np.empty(self._tgt(mode).shape)
            for band_index, sel, rr in self._r:
                assert x.shape[0] == 1
                # FIXME Is ascontiguousarray really a good idea here?
                inp = np.ascontiguousarray(x[0][..., sel])
                res[band_index] = rr.adjoint(ift.makeField(rr.target, inp)).val
                empty[band_index] = False
            for band_index in np.where(empty)[0]:
                # Support empty imaging bands even though this might be a waste
                # of time
                res[band_index] = 0
                empty[band_index] = False
            my_assert(not np.any(empty))
        return ift.makeField(self._tgt(mode), res)


class ResponseDistributor(ift.LinearOperator):
    def __init__(self, *ops):
        dom, tgt = ops[0].domain, ops[0].target
        cap = self.TIMES | self.ADJOINT_TIMES
        for op in ops:
            my_assert_isinstance(op, ift.LinearOperator)
            my_assert(dom is op.domain)
            my_assert(tgt is op.target)
            my_assert(self.TIMES & op.capability, self.ADJOINT_TIMES & op.capability)
        self._domain = ift.makeDomain(dom)
        self._target = ift.makeDomain((ift.UnstructuredDomain(len(ops)), *tgt))
        self._capability = cap
        self._ops = ops

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = []
            for op in self._ops:
                res.append(op(x).val)
            res = np.array(res)
            return ift.makeField(self._tgt(mode), np.array(res))
        for ii, op in enumerate(self._ops):
            new = op.adjoint(ift.makeField(self._ops[0].target, x.val[ii]))
            if ii == 0:
                res = new
            else:
                res = res + new
        return res


class FullResponse(ift.LinearOperator):
    def __init__(self, observation, sky_domain):
        raise NotImplementedError


class SingleResponse(ift.LinearOperator):
    def __init__(self, domain, uvw, freq, mask, single_precision):
        # FIXME Currently only the response uses single_precision if possible.
        # Could be rolled out to the whole likelihood
        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.makeDomain(
            ift.UnstructuredDomain(ss) for ss in (uvw.shape[0], freq.size)
        )
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._args = {
            "uvw": uvw,
            "freq": freq,
            "mask": mask.astype(np.uint8),
            "pixsize_x": self._domain[0].distances[0],
            "pixsize_y": self._domain[0].distances[1],
            "epsilon": epsilon(),
            "do_wgridding": wgridding(),
            "nthreads": nthreads(),
            "flip_v": True,
        }
        self._vol = self._domain[0].scalar_dvol
        self._target_dtype = np.complex64 if single_precision else np.complex128
        self._domain_dtype = np.float32 if single_precision else np.float64
        self._verbt, self._verbadj = True, True

    def apply(self, x, mode):
        self._check_input(x, mode)
        # FIXME mtr Is the sky in single precision mode single or double?
        # my_asserteq(x.dtype, self._domain_dtype if mode == self.TIMES else self._target_dtype)
        x = x.val.astype(
            self._domain_dtype if mode == self.TIMES else self._target_dtype
        )
        if mode == self.TIMES:
            args1 = {"dirty": x}
            if self._verbt:
                args1["verbosity"] = True
                self._verbt = False
            f = dirty2vis
            # FIXME Use vis_out keyword of wgridder
        else:
            # FIXME assert correct strides for visibilities
            my_assert(x.flags["C_CONTIGUOUS"])
            args1 = {
                "vis": x,
                "npix_x": self._domain[0].shape[0],
                "npix_y": self._domain.shape[1],
            }
            if self._verbadj:
                args1["verbosity"] = True
                self._verbadj = False
            f = vis2dirty
        res = ift.makeField(self._tgt(mode), f(**self._args, **args1) * self._vol)
        my_asserteq(
            res.dtype, self._target_dtype if mode == self.TIMES else self._domain_dtype
        )
        return res
