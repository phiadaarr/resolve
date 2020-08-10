# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import h5py
import numpy as np

from .direction import Direction
from .polarization import Polarization
from .util import compare_attributes, my_assert, my_asserteq


class Observation:
    def __init__(self, uvw, vis, weight, polarization, freq, direction):
        nrows = uvw.shape[0]
        my_assert(isinstance(direction, Direction))
        my_assert(isinstance(polarization, Polarization))
        my_asserteq(weight.shape, vis.shape)
        my_asserteq(vis.shape, (len(polarization), nrows, len(freq)))

        self._uvw = uvw
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
        shp0 = self._vis.size
        # Remove flagged rows
        sel = np.any(self._weight != 0, axis=2)
        sel = np.any(sel != 0, axis=0)
        uvw = self._uvw[sel]
        weight = self._weight[:, sel]
        vis = self._vis[:, sel]
        # Remove flagged freqencies
        sel = np.any(weight != 0, axis=0)
        sel = np.any(sel != 0, axis=0)
        vis = vis[..., sel]
        weight = weight[..., sel]
        freq = self._freq[sel]
        print(f'Compression: {shp0} -> {vis.shape}')
        uvw = np.ascontiguousarray(uvw)
        vis = np.ascontiguousarray(vis)
        weight = np.ascontiguousarray(weight)
        return Observation(uvw, vis, weight, self._polarization, freq,
                           self._direction)

    def save_to_hdf5(self, file_name):
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('uvw', data=self._uvw)
            f.create_dataset('vis', data=self._vis)
            f.create_dataset('weight', data=self._weight)
            f.create_dataset('freq', data=self._freq)
            f.create_dataset('polarization', data=self._polarization.to_list())
            f.create_dataset('direction', data=self._direction.to_list())

    @staticmethod
    def load_from_hdf5(file_name):
        with h5py.File(file_name, 'r') as f:
            uvw = np.array(f['uvw'])
            vis = np.array(f['vis'])
            weight = np.array(f['weight'])
            polarization = list(f['polarization'])
            freq = np.array(f['freq'])
            direction = list(f['direction'])
        return Observation(uvw, vis, weight, Polarization.from_list(polarization), freq, Direction.from_list(direction))

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        return compare_attributes(self, other, ('_direction', '_polarization', '_freq', '_uvw', '_vis', '_weight'))

    def average_stokes_i(self):
        inds = self._polarization.stokes_i_indices()
        vis = 0.5*np.sum(self._vis[inds], axis=0)[None]
        weight = 1/(0.5*np.sum(1/self._weight[inds], axis=0))[None]
        return Observation(self._uvw, vis, weight, Polarization.trivial(),
                           self._freq, self._direction)

    def restrict_to_stokes_i(self):
        inds = self._polarization.stokes_i_indices()
        vis = self._vis[inds]
        weight = self._weight[inds]
        pol = self._polarization.restrict_to_stokes_i()
        return Observation(self._uvw, vis, weight, pol, self._freq,
                           self._direction)

    @property
    def uvw(self):
        return self._uvw

    @property
    def vis(self):
        return self._vis

    @property
    def weight(self):
        return self._weight

    @property
    def freq(self):
        return self._freq

    @property
    def polarization(self):
        return self._polarization

    @property
    def direction(self):
        return self._direction
