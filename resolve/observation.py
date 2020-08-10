# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import h5py
import numpy as np

from .direction import Direction
from .util import my_assert


class Observation:
    def __init__(self, uvw, vis, weight, flags, polarization, freq, direction):
        nrows = uvw.shape[0]
        my_assert(isinstance(direction, Direction))
        my_assert(nrows == vis.shape[0])
        my_assert(flags.shape == vis.shape)
        my_assert(len(freq) == vis.shape[1])
        my_assert(len(polarization) == vis.shape[2])
        # Fallunterscheidung weight weightspectrum
        # spw = t.getcol("DATA_DESC_ID")

    def save_to_hdf5(self, file_name):
        raise NotImplementedError

    @staticmethod
    def load_from_hdf5(self, file_name):
        raise NotImplementedError
