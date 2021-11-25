# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np

from .data.observation import Observation
from .irg_space import IRGSpace
from .response_new import SingleResponse
from .util import my_assert, my_assert_isinstance, my_asserteq


def StokesIResponse(observation, domain):
    if isinstance(observation, dict):
        # TODO Possibly add subpixel offset here
        d = ift.MultiDomain.make(domain)
        res = (
            StokesIResponse(o, d[kk]).ducktape(kk).ducktape_left(kk)
            for kk, o in observation.items()
        )
        return reduce(add, res)
    my_assert_isinstance(observation, Observation)
    domain = ift.DomainTuple.make(domain)
    my_asserteq(len(domain), 1)
    my_assert_isinstance(domain[0], ift.RGSpace)
    npol = observation.npol
    my_assert(npol in [1, 2])
    mask = observation.mask.val
    sr0 = SingleResponse(domain, observation.uvw, observation.freq, mask[0])
    # FIXME Get rid of npol==2 support here
    if npol == 1 or (npol == 2 and np.all(mask[0] == mask[1])):
        contr = ift.ContractionOperator(observation.vis.domain, 0)
        return contr.adjoint @ sr0
    elif npol == 2:
        sr1 = SingleResponse(domain, observation.uvw, observation.freq, mask[1])
        return ResponseDistributor(observation.polarization.space, sr0, sr1)
    raise RuntimeError


class FullPolResponse(ift.LinearOperator):
    def __init__(self, observation, domain):
        my_assert_isinstance(observation, Observation)
        domain = ift.MultiDomain.make(domain)
        self._domain = domain
        self._target = observation.vis.domain
        if set(domain.keys()) == set(["I", "Q", "U", "V"]):
            self._with_v = True
        elif set(domain.keys()) == set(["I", "Q", "U"]):
            self._with_v = False
        else:
            raise RuntimeError
        my_asserteq(len(domain["I"]), 1)
        my_asserteq(len(domain["I"].shape), 2)
        my_asserteq(dd for dd in domain)
        npol = observation.npol
        my_asserteq(npol, 4)
        domain = domain["I"]
        mask = np.all(observation.mask.val, axis=0)
        self._sr = SingleResponse(domain, observation.uvw, observation.freq, mask)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.empty(self._target.shape, dtype=np.complex64)
            Q = self._sr(x["Q"]).val
            assert Q.dtype == np.complex64
            U = self._sr(x["U"]).val
            res[0] = res[3] = self._sr(x["I"]).val
            if self._with_v:
                V = self._sr(x["V"]).val
                res[0] += V
                res[3] -= -V
            res[1] = Q + 1j * U
            res[2] = Q - 1j * Q
        else:
            op = lambda inp: self._sr.adjoint(ift.makeField(self._sr.target, inp))
            x = x.val
            res = {}
            res["I"] = op(x[0] + x[3]).val
            if self._with_v:
                res["V"] = op(x[0] - x[3]).val
            res["Q"] = op(x[1] + x[2]).val
            res["U"] = op(1j * (x[2] - x[1])).val
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
        Contains the :class:`IRGSpace` for the frequencies. The coordinates of
        the space can be either frequencies or normalized log-frequencies. In
        the latter case set `log_freq` to True.
    position_domain : nifty8.RGSpace
        Contains the the :class:`nifty8.RGSpace` for the positions.
    log_freq : bool
        If true, the coordinates of `frequency_domain` are interpreted to be
        normalized log-frequencies.  Default is False.
    """

    def __init__(self, observation, frequency_domain, position_domain, log_freq=False):
        my_assert_isinstance(observation, Observation)
        # FIXME Add polarization support
        my_assert(observation.npol in [1, 2])
        my_assert_isinstance(frequency_domain, IRGSpace)
        my_assert_isinstance(position_domain, ift.RGSpace)

        domain_tuple = (frequency_domain, position_domain)
        self._domain = ift.DomainTuple.make(domain_tuple)
        self._target = observation.vis.domain
        self._capability = self.TIMES | self.ADJOINT_TIMES

        if log_freq:
            data_freq = np.log(observation.freq / observation.freq.mean())
        else:
            data_freq = observation.freq

        my_assert(np.all(np.diff(data_freq) > 0))
        sky_freq = np.array(frequency_domain.coordinates)
        band_indices = self.band_indices(sky_freq, data_freq)
        # Make sure that no index is wasted
        my_asserteq(len(set(band_indices)), frequency_domain.size)
        self._r = []
        mask = observation.mask.val
        for band_index in np.unique(band_indices):
            sel = band_indices == band_index
            if mask.shape[0] == 1:
                mymask = mask[0, :, sel]
            elif mask.shape[0] == 2:
                # FIXME In stokesi mode: mask everything possible in gridder, the rest afterwards.
                mymask = np.any(mask[0, :, sel], axis=0)
            else:
                raise NotImplementedError
            r = SingleResponse(
                position_domain,
                observation.uvw,
                observation.freq[sel],
                mymask.T,
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
                res[..., sel] = foo.val
        else:
            empty = np.zeros(self._domain.shape[0], bool)
            res = np.empty(self._tgt(mode).shape)
            for band_index, sel, rr in self._r:
                assert x.shape[0] == 1
                # FIXME Is ascontiguousarray really a good idea here?
                inp = np.ascontiguousarray(np.sum(x[..., sel], axis=0))
                res[band_index] = rr.adjoint(ift.makeField(rr.target, inp)).val
                empty[band_index] = False
            for band_index in np.where(empty)[0]:
                # Support empty imaging bands even though this might be a waste
                # of time
                res[band_index] = 0
                empty[band_index] = False
            my_assert(not np.any(empty))
        return ift.makeField(self._tgt(mode), res)

    @staticmethod
    def band_indices(sky_freq, data_freq):
        return [np.argmin(np.abs(ff - sky_freq)) for ff in data_freq]


class ResponseDistributor(ift.LinearOperator):
    def __init__(self, iterating_target_space, *ops):
        dom, tgt = ops[0].domain, ops[0].target
        cap = self.TIMES | self.ADJOINT_TIMES
        for op in ops:
            my_assert_isinstance(op, ift.LinearOperator)
            my_assert(dom is op.domain)
            my_assert(tgt is op.target)
            my_assert(self.TIMES & op.capability, self.ADJOINT_TIMES & op.capability)
        my_asserteq(len(iterating_target_space.shape), 1)
        my_asserteq(len(ops), iterating_target_space.shape[0])
        self._domain = ift.makeDomain(dom)
        self._target = ift.makeDomain((iterating_target_space, *tgt))
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
