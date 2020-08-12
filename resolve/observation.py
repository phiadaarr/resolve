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
from .util import compare_attributes, my_assert_isinstance, my_asserteq


# FIXME Introduce Observation.nfreqs, .nrows, .npol

class Observation:
    def __init__(self, antenna_positions, vis, weight, polarization, freq, direction):
        nrows = len(antenna_positions)
        my_assert_isinstance(direction, Direction)
        my_assert_isinstance(polarization, Polarization)
        my_assert_isinstance(antenna_positions, AntennaPositions)
        my_asserteq(weight.shape, vis.shape)
        my_asserteq(vis.shape, (len(polarization), nrows, len(freq)))
        my_asserteq(nrows, vis.shape[1])

        self._antpos = antenna_positions
        self._vis = vis
        self._weight = weight
        self._polarization = polarization
        self._freq = freq
        self._direction = direction

    def max_snr(self):
        inds = self._weight > 0
        return np.max(np.abs(self._vis[inds])*np.sqrt(self._weight[inds]))

    def fraction_flagged(self):
        return np.sum(self._weight == 0)/self._weight.size

    def compress(self):
        shp0 = self._vis.shape
        # TODO Iterate between rows and frequencies until nothing can be removed anymore
        # Remove flagged rows
        sel = np.any(self._weight != 0, axis=2)
        sel = np.any(sel != 0, axis=0)
        antpos = self._antpos[sel]
        weight = self._weight[:, sel]
        vis = self._vis[:, sel]
        # Remove flagged freqencies
        sel = np.any(weight != 0, axis=0)
        sel = np.any(sel != 0, axis=0)
        vis = vis[..., sel]
        weight = weight[..., sel]
        freq = self._freq[sel]
        print(f'Compression: {shp0} -> {vis.shape}')
        vis = np.ascontiguousarray(vis)
        weight = np.ascontiguousarray(weight)
        return Observation(antpos, vis, weight, self._polarization, freq,
                           self._direction)

    def save_to_hdf5(self, file_name):
        with h5py.File(file_name, 'w') as f:
            for ii, vv in enumerate(self._antpos.to_list()):
                f.create_dataset(f'antenna_positions{ii}', data=vv)
            f.create_dataset('vis', data=self._vis)
            f.create_dataset('weight', data=self._weight)
            f.create_dataset('freq', data=self._freq)
            f.create_dataset('polarization', data=self._polarization.to_list())
            f.create_dataset('direction', data=self._direction.to_list())

    @staticmethod
    def load_from_hdf5(file_name):
        with h5py.File(file_name, 'r') as f:
            antpos = [np.array(f[f'antenna_positions{ii}']) for ii in range(4)]
            vis = np.array(f['vis'])
            weight = np.array(f['weight'])
            polarization = list(f['polarization'])
            freq = np.array(f['freq'])
            direction = list(f['direction'])
        return Observation(AntennaPositions.from_list(antpos), vis, weight, Polarization.from_list(polarization), freq, Direction.from_list(direction))

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        return compare_attributes(self, other, ('_direction', '_polarization', '_freq', '_antpos', '_vis', '_weight'))

    def average_stokes_i(self):
        inds = self._polarization.stokes_i_indices()
        vis = 0.5*np.sum(self._vis[inds], axis=0)[None]
        weight = 1/(0.5*np.sum(1/self._weight[inds], axis=0))[None]
        return Observation(self._antpos, vis, weight, Polarization.trivial(),
                           self._freq, self._direction)

    def restrict_to_stokes_i(self):
        inds = self._polarization.stokes_i_indices()
        vis = self._vis[inds]
        weight = self._weight[inds]
        pol = self._polarization.restrict_to_stokes_i()
        return Observation(self._antpos, vis, weight, pol, self._freq,
                           self._direction)

    def move_time(self, t0):
        antpos = self._antpos.move_time(t0)
        return Observation(antpos, self._vis, self._weight, self._polarization,
                           self._freq, self._direction)

    @property
    def uvw(self):
        return self._antpos.uvw

    @property
    def antenna_positions(self):
        return self._antpos

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
    def freq(self):
        return self._freq

    @property
    def polarization(self):
        return self._polarization

    @property
    def direction(self):
        return self._direction


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
