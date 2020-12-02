# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

from .antenna_positions import AntennaPositions
from .constants import SPEEDOFLIGHT
from .direction import Direction
from .mpi import onlymaster
from .polarization import Polarization
from .util import compare_attributes, my_assert, my_assert_isinstance, my_asserteq


class Observation:
    """Observation data

    This class contains all the data and information about an observation.
    It supports a single field (phase center) and a single spectral window.

    Parameters
    ----------
    antenna_positions : AntennaPositions
        Instance of the :class:`AntennaPositions` that contains all information on antennas and baselines.
    vis : numpy.ndarray
        Contains the measured visibilities. Shape (n_polarizations, n_rows, n_channels)
    weight : numpy.ndarray
        Contains the information from the WEIGHT or SPECTRUM_WEIGHT column.
        This is in many cases the inverse of the thermal noise covariance. Shape same as vis.
    polarization : Polarization
    freq : numpy.ndarray
        Contains the measured frequencies. Shape (n_channels)
    direction : Direction

    Note
    ----
    vis and weight must have the same dimensions
    """

    def __init__(self, antenna_positions, vis, weight, polarization, freq, direction):
        nrows = len(antenna_positions)
        my_assert_isinstance(direction, Direction)
        my_assert_isinstance(polarization, Polarization)
        my_assert_isinstance(antenna_positions, AntennaPositions)
        my_asserteq(weight.shape, vis.shape)
        my_asserteq(vis.shape, (len(polarization), nrows, len(freq)))
        my_asserteq(nrows, vis.shape[1])
        my_assert(np.all(weight >= 0.0))

        vis.flags.writeable = False
        weight.flags.writeable = False

        self._antpos = antenna_positions
        self._vis = vis
        self._weight = weight
        self._polarization = polarization
        self._freq = freq
        self._direction = direction

    def apply_flags(self, arr):
        return arr[self._weight != 0.0]

    @property
    def flags(self):
        return self._weight == 0.0

    @property
    def mask(self):
        return self._weight > 0.0

    def max_snr(self):
        return np.max(np.abs(self.apply_flags(self._vis * np.sqrt(self._weight))))

    def fraction_useful(self):
        return self.apply_flags(self._weight).size / self._weight.size

    @onlymaster
    def save(self, file_name, compress):
        dct = dict(
            vis=self._vis,
            weight=self._weight,
            freq=self._freq,
            polarization=self._polarization.to_list(),
            direction=self._direction.to_list(),
        )
        for ii, vv in enumerate(self._antpos.to_list()):
            if vv is None:
                vv = np.array([])
            dct[f"antpos{ii}"] = vv
        f = np.savez_compressed if compress else np.savez
        f(file_name, **dct)

    @staticmethod
    def load(file_name):
        dct = dict(np.load(file_name))
        antpos = []
        for ii in range(4):
            val = dct[f"antpos{ii}"]
            if val.size == 0:
                val = None
            antpos.append(val)
        pol = Polarization.from_list(dct["polarization"])
        direction = Direction.from_list(dct["direction"])
        return Observation(
            AntennaPositions.from_list(antpos),
            dct["vis"],
            dct["weight"],
            pol,
            dct["freq"],
            direction,
        )

    def __eq__(self, other):
        if not isinstance(other, Observation):
            return False
        if (
            self._vis.dtype != other._vis.dtype
            or self._weight.dtype != other._weight.dtype
        ):
            return False
        return compare_attributes(
            self,
            other,
            ("_direction", "_polarization", "_freq", "_antpos", "_vis", "_weight"),
        )

    def __getitem__(self, slc):
        return Observation(
            self._antpos[slc],
            self._vis[:, slc],
            self._weight[:, slc],
            self._polarization,
            self._freq,
            self._direction,
        )

    def delete_antenna_information(self):
        raise NotImplementedError

    def move_time(self, t0):
        antpos = self._antpos.move_time(t0)
        return Observation(
            antpos,
            self._vis,
            self._weight,
            self._polarization,
            self._freq,
            self._direction,
        )

    @property
    def uvw(self):
        return self._antpos.uvw

    @property
    def antenna_positions(self):
        return self._antpos

    def effective_uvw(self):
        out = np.einsum("ij,k->ijk", self.uvw, self._freq / SPEEDOFLIGHT)
        my_asserteq(out.shape, (self.nrow, 3, self.nfreq))
        return out

    def effective_uv(self):
        out = np.einsum("ij,k->ijk", self.uvw[:, 0:2], self._freq / SPEEDOFLIGHT)
        my_asserteq(out.shape, (self.nrow, 2, self.nfreq))
        return out

    def effective_uvwlen(self):
        uvlen = np.linalg.norm(self.uvw, axis=1)
        return np.outer(uvlen, self._freq / SPEEDOFLIGHT)

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
    """

    Parameters
    ----------
    args : Observation or list of Observation

    Returns
    -------
    mi, ma : tuple of float
        first and last measurement time point

    """
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
