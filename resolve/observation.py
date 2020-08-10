# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import h5py
import numpy as np

from .direction import Direction
from .polarization import Polarization
from .util import compare_attributes, my_assert


class Observation:
    def __init__(self, uvw, vis, weight, flags, polarization, freq, direction):
        nrows = uvw.shape[0]
        my_assert(isinstance(direction, Direction))
        my_assert(isinstance(polarization, Polarization))
        my_assert(nrows == vis.shape[0])
        my_assert(flags.shape == vis.shape)
        my_assert(len(freq) == vis.shape[1])
        my_assert(len(polarization) == vis.shape[2])
        # Fallunterscheidung weight weightspectrum

        self._uvw = uvw
        self._vis = vis
        self._weight = weight
        self._flags = flags
        self._polarization = polarization
        self._freq = freq
        self._direction = direction

    def save_to_hdf5(self, file_name):
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('uvw', data=self._uvw)
            f.create_dataset('vis', data=self._vis)
            f.create_dataset('weight', data=self._weight)
            f.create_dataset('flags', data=self._flags)
            f.create_dataset('freq', data=self._freq)
            f.create_dataset('polarization', data=self._polarization.to_list())
            f.create_dataset('direction', data=self._direction.to_list())

    @staticmethod
    def load_from_hdf5(file_name):
        with h5py.File(file_name, 'r') as f:
            uvw = np.array(f['uvw'])
            vis = np.array(f['vis'])
            weight = np.array(f['weight'])
            flags = np.array(f['flags'])
            polarization = list(f['polarization'])
            freq = np.array(f['freq'])
            direction = list(f['direction'])
        return Observation(uvw, vis, weight, flags, Polarization.from_list(polarization), freq, Direction.from_list(direction))

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        return compare_attributes(self, other, ('_direction', '_polarization', '_freq', '_flags', '_uvw', '_vis', '_weight'))
