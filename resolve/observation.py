# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty7 as ift

from .antenna_positions import AntennaPositions
from .constants import SPEEDOFLIGHT
from .direction import Direction, Directions
from .mpi import onlymaster
from .polarization import Polarization
from .util import compare_attributes, my_assert, my_assert_isinstance, my_asserteq


class _Observation:
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


class SingleDishObservation(_Observation):
    def __init__(self, pointings, data, weight, polarization, freq):
        my_assert_isinstance(pointings, Directions)
        my_assert_isinstance(polarization, Polarization)
        my_assert(data.dtype in [np.float32, np.float64])
        nrows = len(pointings)
        my_asserteq(weight.shape, data.shape)
        my_asserteq(data.shape, (len(polarization), nrows, len(freq)))
        my_asserteq(nrows, data.shape[1])

        data.flags.writeable = False
        weight.flags.writeable = False

        my_assert(np.all(weight >= 0.0))
        my_assert(np.all(np.isfinite(data)))
        my_assert(np.all(np.isfinite(weight)))

        self._pointings = pointings
        self._vis = data
        self._weight = weight
        self._polarization = polarization
        self._freq = freq

    @onlymaster
    def save(self, file_name, compress):
        p = self._pointings.to_list()
        dct = dict(
            vis=self._vis,
            weight=self._weight,
            freq=self._freq,
            polarization=self._polarization.to_list(),
            pointings0=p[0],
            pointings1=p[1],
        )
        f = np.savez_compressed if compress else np.savez
        f(file_name, **dct)

    @staticmethod
    def load(file_name):
        dct = dict(np.load(file_name))
        pol = Polarization.from_list(dct["polarization"])
        pointings = Directions.from_list([dct["pointings0"], dct["pointings1"]])
        return SingleDishObservation(
            pointings, dct["vis"], dct["weight"], pol, dct["freq"]
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
            self, other, ("_polarization", "_freq", "_pointings", "_vis", "_weight")
        )

    def __getitem__(self, slc):
        return SingleDishObservation(
            self._pointings[slc],
            self._vis[:, slc],
            self._weight[:, slc],
            self._polarization,
            self._freq,
        )

    @property
    def pointings(self):
        return self._pointings


class Observation(_Observation):
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
        my_assert(np.all(np.isfinite(vis)))
        my_assert(np.all(np.isfinite(weight)))

        vis.flags.writeable = False
        weight.flags.writeable = False

        self._antpos = antenna_positions
        self._vis = vis
        self._weight = weight
        self._polarization = polarization
        self._freq = freq
        self._direction = direction
        self._ndeff = None

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
        return self.n_data_effective / self._weight.size

    @property
    def n_data_effective(self):
        if self._ndeff is None:
            self._ndeff = self.apply_flags(self._weight).size
        return self._ndeff

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
    def load(file_name, lo_hi_index=None):
        dct = dict(np.load(file_name))
        antpos = []
        for ii in range(4):
            val = dct[f"antpos{ii}"]
            if val.size == 0:
                val = None
            antpos.append(val)
        pol = Polarization.from_list(dct["polarization"])
        direction = Direction.from_list(dct["direction"])
        slc = slice(None) if lo_hi_index is None else slice(*lo_hi_index)
        # FIXME Put barrier here that makes sure that only one full Observation is loaded at a time
        return Observation(
            AntennaPositions.from_list(antpos),
            dct["vis"][..., slc],
            dct["weight"][..., slc],
            pol,
            dct["freq"][slc],
            direction,
        )

    @staticmethod
    def load_mf(file_name, n_imaging_bands, comm=None):
        if comm is not None:
            my_assert(n_imaging_bands >= comm.Get_size())

        # Compute frequency ranges in data space
        global_freqs = np.load(file_name).get("freq")
        assert np.all(np.diff(global_freqs) > 0)
        my_assert(n_imaging_bands <= global_freqs.size)

        if comm is None:
            local_imaging_bands = range(n_imaging_bands)
        else:
            local_imaging_bands = range(
                *ift.utilities.shareRange(
                    n_imaging_bands, comm.Get_size(), comm.Get_rank()
                )
            )
        full_obs = Observation.load(file_name)
        obs_list = [
            full_obs.get_freqs_by_slice(
                slice(*ift.utilities.shareRange(len(global_freqs), n_imaging_bands, ii))
            )
            for ii in local_imaging_bands
        ]
        nu0 = global_freqs.mean()
        return obs_list, nu0

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

    def get_freqs(self, frequency_list):
        """Return observation that contains a subset of the present frequencies

        Parameters
        ----------
        frequency_list : list
            List of indices that shall be returned
        """
        mask = np.zeros(self.nfreq, dtype=bool)
        mask[frequency_list] = 1
        return self.get_freqs_by_slice(mask)

    def get_freqs_by_slice(self, slc):
        return Observation(
            self._antpos,
            self._vis[..., slc],
            self._weight[..., slc],
            self._polarization,
            self._freq[slc],
            self._direction,
        )

    def average_stokesi(self):
        my_asserteq(self._vis.shape[0], 2)
        my_asserteq(self._polarization.restrict_to_stokes_i(), self._polarization)
        vis = np.sum(self._weight * self._vis, axis=0)[None]
        wgt = np.sum(self._weight, axis=0)[None]
        invmask = wgt == 0.0
        vis /= wgt + np.ones_like(wgt) * invmask
        vis[invmask] = 0.0
        return Observation(
            self._antpos, vis, wgt, Polarization.trivial(), self._freq, self._direction
        )

    def restrict_to_stokesi(self):
        my_asserteq(self._vis.shape[0], 4)
        ind = self._polarization.stokes_i_indices()
        vis = self._vis[ind]
        wgt = self._weight[ind]
        pol = self._polarization.restrict_to_stokes_i()
        return Observation(self._antpos, vis, wgt, pol, self._freq, self._direction)

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
        return np.outer(self.uvwlen(), self._freq / SPEEDOFLIGHT)

    def uvwlen(self):
        return np.linalg.norm(self.uvw, axis=1)


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
