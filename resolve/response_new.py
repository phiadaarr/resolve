# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras, Jakob Knollmüller

from itertools import product

import nifty8 as ift
import numpy as np
from ducc0.wgridder.experimental import dirty2vis, vis2dirty

from .constants import SPEEDOFLIGHT
from .data.observation import Observation
from .global_config import epsilon, np_dtype, nthreads, verbosity, wgridding
from .util import (assert_sky_domain, my_assert, my_assert_isinstance,
                   my_asserteq)


class InterferometryResponse(ift.LinearOperator):
    def __init__(self, observation, domain):
        assert isinstance(observation, Observation)
        domain = ift.DomainTuple.make(domain)
        assert_sky_domain(domain)

        pdom, tdom, fdom, sdom = domain

        n_time_bins = tdom.shape[0]
        n_freq_bins = fdom.shape[0]

        t_binbounds = tdom.binbounds()
        f_binbounds = fdom.binbounds()

        sr = []
        row_indices, freq_indices = [], []
        for tt in range(n_time_bins):
            sr_tmp, t_tmp, f_tmp = [], [], []
            if tuple(t_binbounds[tt:tt+2]) == (-np.inf, np.inf):
                oo = observation
                tind = slice(None)
            else:
                oo, tind = observation.restrict_by_time(t_binbounds[tt], t_binbounds[tt+1], True)
            for ff in range(n_freq_bins):
                ooo, find = oo.restrict_by_freq(f_binbounds[ff], f_binbounds[ff+1], True)
                if any(np.array(ooo.vis.shape) == 0):
                    rrr = None
                else:
                    rrr = SingleResponse(domain[3], ooo.uvw, ooo.freq)   # FIXME Future: Include mask here?
                sr_tmp.append(rrr)
                t_tmp.append(tind)
                f_tmp.append(find)
            sr.append(sr_tmp)
            row_indices.append(t_tmp)
            freq_indices.append(f_tmp)
        self._sr = sr
        self._row_indices = row_indices
        self._freq_indices = freq_indices

        self._domain = domain
        self._target = ift.makeDomain((domain[0],) + observation.vis.domain[1:])
        self._capability = self.TIMES | self.ADJOINT_TIMES

        foo = np.zeros(self.target.shape, np.int8)
        for pp in range(self.domain.shape[0]):
            for tt in range(self.domain.shape[1]):
                for ff in range(self.domain.shape[2]):
                    foo[pp, self._row_indices[tt][ff], self._freq_indices[tt][ff]] = 1.
        if np.any(foo == 0):
            raise RuntimeError("This should not happen. This is a bug in `InterferometryResponse`.")

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = self._times(x)
        else:
            res = self._adj_times(x)
        return ift.makeField(self._tgt(mode), res)

    def _times(self, x):
        res = np.empty(self.target.shape, np.complex128)
        for pp in range(self.domain.shape[0]):
            for tt in range(self.domain.shape[1]):
                for ff in range(self.domain.shape[2]):
                    op = self._sr[tt][ff]
                    if op is None:
                        continue
                    inp = ift.makeField(op.domain, x[pp, tt, ff])
                    res[pp, self._row_indices[tt][ff], self._freq_indices[tt][ff]] = op(inp).val
        return res

    def _adj_times(self, x):
        res = np.zeros(self.domain.shape, np.float64)
        for pp in range(self.domain.shape[0]):
            for tt in range(self.domain.shape[1]):
                for ff in range(self.domain.shape[2]):
                    op = self._sr[tt][ff]
                    if op is None:
                        continue
                    inp = x[pp, self._row_indices[tt][ff], self._freq_indices[tt][ff]]
                    inp = ift.makeField(op.target, inp)
                    res[pp, tt, ff] = op.adjoint(inp).val
        return res


class SingleResponse(ift.LinearOperator):
    def __init__(self, domain, uvw, freq, mask=None, facets=(1, 1)):
        my_assert_isinstance(facets, tuple)
        for ii in range(2):
            if domain.shape[ii] % facets[ii] != 0:
                raise ValueError("nfacets needs to be divisor of npix.")
        self._domain = ift.DomainTuple.make(domain)
        tgt = ift.UnstructuredDomain(uvw.shape[0]), ift.UnstructuredDomain(freq.size)
        self._target = ift.makeDomain(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._args = {
            "uvw": uvw,
            "freq": freq,
            "pixsize_x": self._domain[0].distances[0],
            "pixsize_y": self._domain[0].distances[1],
            "epsilon": epsilon(),
            "do_wgridding": wgridding(),
            "nthreads": nthreads(),
            "flip_v": True,
        }
        if mask is not None:
            self._args["mask"] = mask.astype(np.uint8)
        self._vol = self._domain[0].scalar_dvol
        self._target_dtype = np_dtype(True)
        self._domain_dtype = np_dtype(False)
        self._ofac = None
        self._facets = facets

    def apply(self, x, mode):
        self._check_input(x, mode)
        one_facet = self._facets == (1, 1)
        x = x.val
        if mode == self.TIMES:
            x = x.astype(self._domain_dtype, copy=False)
            if one_facet:
                res = self._times(x)
            else:
                res = self._facet_times(x)
        else:
            x = x.astype(self._target_dtype, copy=False)
            if one_facet:
                res = self._adjoint(x)
            else:
                res = self._facet_adjoint(x)
        res = ift.makeField(self._tgt(mode), res * self._vol)

        expected_dtype = self._target_dtype if mode == self.TIMES else self._domain_dtype
        my_asserteq(res.dtype, expected_dtype)

        return res

    def oversampling_factors(self):
        if self._ofac is not None:
            return self._ofac
        maxuv = (
                np.max(np.abs(self._args["uvw"][:, 0:2]), axis=0)
                * np.max(self._args["freq"])
                / SPEEDOFLIGHT
        )
        hspace = self._domain[0].get_default_codomain()
        hvol = np.array(hspace.shape) * np.array(hspace.distances) / 2
        self._ofac = hvol / maxuv
        return self._ofac

    def _times(self, x):
        if verbosity():
            print(f"\nINFO: Oversampling factors in response: {self.oversampling_factors()}\n")
        return dirty2vis(dirty=x, verbosity=verbosity(), **self._args)

    def _adjoint(self, x):
        my_assert(x.flags["C_CONTIGUOUS"])
        nx, ny = self._domain.shape
        if verbosity():
            print(f"\nINFO: Oversampling factors in response: {self.oversampling_factors()}\n")
        return vis2dirty(vis=x, npix_x=nx, npix_y=ny, verbosity=verbosity(), **self._args)

    def _facet_times(self, x):
        nfacets_x, nfacets_y = self._facets
        npix_x, npix_y = self._domain.shape
        nx = npix_x // nfacets_x
        ny = npix_y // nfacets_y
        vis = None
        for xx, yy in product(range(nfacets_x), range(nfacets_y)):
            cx = ((0.5 + xx) / nfacets_x - 0.5) * self._args["pixsize_x"] * npix_x
            cy = ((0.5 + yy) / nfacets_y - 0.5) * self._args["pixsize_y"] * npix_y
            facet = x[nx * xx: nx * (xx + 1), ny * yy: ny * (yy + 1)]
            foo = dirty2vis(
                dirty=facet,
                center_x=cx,
                center_y=cy,
                verbosity=verbosity(),
                **self._args
            )
            if vis is None:
                vis = foo
            else:
                vis += foo
        return vis

    def _facet_adjoint(self, x):
        nfacets_x, nfacets_y = self._facets
        npix_x, npix_y = self._domain.shape
        nx = npix_x // nfacets_x
        ny = npix_y // nfacets_y
        res = np.zeros((npix_x, npix_y), self._domain_dtype)
        for xx, yy in product(range(nfacets_x), range(nfacets_y)):
            cx = ((0.5 + xx) / nfacets_x - 0.5) * self._args["pixsize_x"] * npix_x
            cy = ((0.5 + yy) / nfacets_y - 0.5) * self._args["pixsize_y"] * npix_y
            im = vis2dirty(
                vis=x,
                npix_x=nx,
                npix_y=ny,
                center_x=cx,
                center_y=cy,
                verbosity=verbosity(),
                **self._args
            )
            res[nx * xx: nx * (xx + 1), ny * yy: ny * (yy + 1)] = im
        return res
