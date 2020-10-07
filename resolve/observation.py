# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import h5py
import numpy as np

import nifty7 as ift

from .antenna_positions import AntennaPositions
from .constants import SPEEDOFLIGHT
from .direction import Direction
from .polarization import Polarization
from .util import (compare_attributes, my_assert, my_assert_isinstance,
                   my_asserteq)


class Observation:
    def __init__(self, antenna_positions, vis, weight, flags, polarization, freq, direction):
        nrows = len(antenna_positions)
        my_assert_isinstance(direction, Direction)
        my_assert_isinstance(polarization, Polarization)
        my_assert_isinstance(antenna_positions, AntennaPositions)
        my_asserteq(weight.shape, vis.shape, flags.shape)
        my_asserteq(vis.shape, (len(polarization), nrows, len(freq)))
        my_asserteq(nrows, vis.shape[1])
        my_assert(np.all(weight > 0))

        self._antpos = antenna_positions
        self._vis = vis
        self._weight = weight
        self._flags = flags
        self._polarization = polarization
        self._freq = freq
        self._direction = direction

    def max_snr(self):
        return np.max(np.abs(self._vis[self._flags])*np.sqrt(self._weight[self._flags]))

    def fraction_useful(self):
        return np.sum(self._flags == 0)/self._weight.size

    def compress(self):
        shp0 = self._vis.shape
        # TODO Iterate between rows and frequencies until nothing can be removed anymore
        # Remove flagged rows
        sel = np.any(self._flags != 0, axis=2)
        sel = np.any(sel != 0, axis=0)
        antpos = self._antpos[sel]
        weight = self._weight[:, sel]
        flags = self._flags[:, sel]
        vis = self._vis[:, sel]
        # Remove flagged freqencies
        sel = np.any(flags != 0, axis=0)
        sel = np.any(sel != 0, axis=0)
        vis = vis[..., sel]
        weight = weight[..., sel]
        flags = flags[..., sel]
        freq = self._freq[sel]
        print(f'Compression: {shp0} -> {vis.shape}')
        vis = np.ascontiguousarray(vis)
        weight = np.ascontiguousarray(weight)
        flags = np.ascontiguousarray(flags)
        return Observation(antpos, vis, weight, flags, self._polarization,
                           freq, self._direction)

    def save_to_hdf5(self, file_name):
        with h5py.File(file_name, 'w') as f:
            for ii, vv in enumerate(self._antpos.to_list()):
                f.create_dataset(f'antenna_positions{ii}', data=vv)
            f.create_dataset('vis', data=self._vis)
            f.create_dataset('weight', data=self._weight)
            f.create_dataset('flags', data=self._flags)
            f.create_dataset('freq', data=self._freq)
            f.create_dataset('polarization', data=self._polarization.to_list())
            f.create_dataset('direction', data=self._direction.to_list())

    @staticmethod
    def load_from_hdf5(file_name):
        with h5py.File(file_name, 'r') as f:
            antpos = [np.array(f[f'antenna_positions{ii}']) for ii in range(4)]
            vis = np.array(f['vis'])
            weight = np.array(f['weight'])
            flags = np.array(f['flags'])
            polarization = list(f['polarization'])
            freq = np.array(f['freq'])
            direction = list(f['direction'])
        return Observation(AntennaPositions.from_list(antpos), vis, weight, flags, Polarization.from_list(polarization), freq, Direction.from_list(direction))

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        if self._vis.dtype != other._vis.dtype or self._weight.dtype != other._weight.dtype:
            return False
        return compare_attributes(self, other, ('_direction', '_polarization', '_freq', '_antpos', '_vis', '_weight'))

    def __getitem__(self, slc):
        return Observation(self._antpos[slc], self._vis[:, slc], self._weight[:, slc], self._flags[:, slc], self._polarization, self._freq, self._direction)

    def average_stokes_i(self):
        indL, indR = self._polarization.stokes_i_indices()
        visL, visR = self._vis[indL], self._vis[indR]
        wgtL, wgtR = self._weight[indL], self._weight[indR]
        vis = 0.5*(wgtL*visL+wgtR*visR)/(wgtL+wgtR)
        cov = 0.25*(1/wgtL+1/wgtR)
        weight = 1/cov
        flags = np.all(self._flags[self._polarization.stokes_i_indices()], axis=0)
        vis, weight, flags = vis[None], weight[None], flags[None]
        f = np.ascontiguousarray
        vis, weight, flags = f(vis), f(weight), f(flags)
        return Observation(self._antpos, vis, weight, flags, Polarization.trivial(),
                           self._freq, self._direction)

    def restrict_to_stokes_i(self):
        inds = self._polarization.stokes_i_indices()
        vis = self._vis[inds]
        weight = self._weight[inds]
        flags = self._flags[inds]
        pol = self._polarization.restrict_to_stokes_i()
        f = np.ascontiguousarray
        vis, weight, flags = f(vis), f(weight), f(flags)
        return Observation(self._antpos, vis, weight, flags, pol, self._freq,
                           self._direction)

    def delete_antenna_information(self):
        raise NotImplementedError

    def move_time(self, t0):
        antpos = self._antpos.move_time(t0)
        return Observation(antpos, self._vis, self._weight, self._flags,
                           self._polarization, self._freq, self._direction)

    @property
    def uvw(self):
        return self._antpos.uvw

    @property
    def antenna_positions(self):
        return self._antpos

    def effective_uvw(self):
        out = np.einsum("ij,k->ijk", self.uvw, self._freq/SPEEDOFLIGHT)
        my_asserteq(out.shape, (self.nrow, 3, self.nfreq))
        return out

    def effective_uv(self):
        out = np.einsum("ij,k->ijk", self.uvw[:, 0:2], self._freq/SPEEDOFLIGHT)
        my_asserteq(out.shape, (self.nrow, 2, self.nfreq))
        return out

    def effective_uvwlen(self):
        uvlen = np.linalg.norm(self.uvw, axis=1)
        return np.outer(uvlen, self._freq/SPEEDOFLIGHT)

    @property
    def vis(self):
        dom = [ift.UnstructuredDomain(ss) for ss in self._vis.shape]
        return ift.makeField(dom, self._vis)

    @property
    def weight(self):
        dom = [ift.UnstructuredDomain(ss) for ss in self._weight.shape]
        return ift.makeField(dom, self._weight)

    @property
    def flags(self):
        dom = [ift.UnstructuredDomain(ss) for ss in self._weight.shape]
        return ift.makeField(dom, self._flags)

    @property
    def freq(self):
        return self._freq

    @property
    def polarization(self):
        return self._polarization

    @property
    def direction(self):
        return self._direction

    @property
    def npol(self):
        return self._vis.shape[0]

    @property
    def nrow(self):
        return self._vis.shape[1]

    @property
    def nfreq(self):
        return self._vis.shape[2]


def tmin_tmax(*args):
    my_assert_isinstance(*args, Observation)
    mi = min([np.min(aa.antenna_positions.time) for aa in args])
    ma = max([np.max(aa.antenna_positions.time) for aa in args])
    return mi, ma


def unique_antennas(*args):
    my_assert_isinstance(*args, Observation)
    antennas = set()
    for oo in args:
        antennas = antennas | oo.antenna_positions.unique_antennas()
    return antennas


def unique_times(*args):
    my_assert_isinstance(*args, Observation)
    times = set()
    for oo in args:
        times = times | oo.antenna_positions.unique_times()
    return times
