# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import numpy as np

import nifty8 as ift

from .antenna_positions import AntennaPositions
from ..constants import SPEEDOFLIGHT
from .direction import Direction, Directions
from ..mpi import onlymaster
from .polarization import Polarization
from ..util import compare_attributes, my_assert, my_assert_isinstance, my_asserteq


class BaseObservation:
    @property
    def _dom(self):
        dom = [ift.UnstructuredDomain(ss) for ss in self._vis.shape]
        return ift.makeDomain(dom)

    @property
    def vis(self):
        """nifty8.Field : Field that contains all data points including
        potentially flagged ones.  Shape: `(npol, nrow, nchan)`, dtype: `numpy.complexfloating`."""
        return ift.makeField(self._dom, self._vis)

    @property
    def weight(self):
        """nifty8.Field : Field that contains all weights, i.e. the diagonal of
        the inverse covariance. Shape: `(npol, nrow, nchan)`, dtype: `numpy.floating`.

        Note
        ----
        If an entry equals 0, this means that the corresponding data point is
        supposed to be ignored.
        """
        return ift.makeField(self._dom, self._weight)

    @property
    def freq(self):
        """numpy.ndarray: One-dimensional array that contains the observing
        frequencies. Shape: `(nchan,), dtype: `np.float64`."""
        return self._freq

    @property
    def polarization(self):
        """Polarization: Object that contains polarization information on the
        data set."""
        return self._polarization

    @property
    def direction(self):
        """Direction: Object that contains direction information on the data
        set."""
        return self._direction

    @property
    def npol(self):
        """int: Number of polarizations present in the data set."""
        return self._vis.shape[0]

    @property
    def nrow(self):
        """int: Number of rows in the data set."""
        return self._vis.shape[1]

    @property
    def nfreq(self):
        """int: Number of observing frequencies."""
        return self._vis.shape[2]

    def apply_flags(self, field):
        """Apply flags to a given field.

        Parameters
        ----------
        field: nifty8.Field
            The field that is supposed to be flagged.

        Returns
        -------
        nifty8.Field
            Flagged field defined on a one-dimensional
            `nifty8.UnstructuredDomain`."""
        return ift.MaskOperator(self.flags)(field)

    @property
    def flags(self):
        """nifty8.Field: True for bad visibilities. May be used together with
        `ift.MaskOperator`."""
        return ift.makeField(self._dom, self._weight == 0.0)

    @property
    def mask(self):
        """nifty8.Field: True for good visibilities."""
        return ift.makeField(self._dom, self._weight > 0.0)

    @property
    def mask_operator(self):
        """nifty8.MaskOperator: Nifty operator that can be used to extract all
        non-flagged data points from a field defined on `self.vis.domain`."""
        return ift.MaskOperator(self.flags)

    def flags_to_nan(self):
        raise NotImplementedError

    def max_snr(self):
        """float: Maximum signal-to-noise ratio."""
        snr = (self.vis * self.weight.sqrt()).abs()
        snr = self.apply_flags(snr)
        return np.max(snr.val)

    def fraction_useful(self):
        """float: Fraction of non-flagged data points."""
        return self.n_data_effective() / self._dom.size

    def n_data_effective(self):
        """int: Number of effective (i.e. non-flagged) data points."""
        return self.mask.s_sum()

    @onlymaster
    def save(self, file_name, compress):
        """Save observation object to disk

        Counterpart to :meth:`load`.

        Parameters
        ----------
        file_name : str
            File name of output file
        compress : bool
            Determine if output file shall be compressed or not. The compression
            algorithm built into numpy is used for this.

        Note
        ----
        If MPI is enabled, this function is only executed on the master task.
        """
        return NotImplementedError

    @staticmethod
    def load(file_name):
        """Load observation object from disk

        Counterpart to :meth:`save`.

        Parameters
        ----------
        file_name : str
            File name of the input file
        """
        return NotImplementedError

    def __getitem__(self, slc):
        return NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if (
                self._vis.dtype != other._vis.dtype
                or self._weight.dtype != other._weight.dtype
        ):
            return False
        return compare_attributes(self, other, self._eq_attributes)


class SingleDishObservation(BaseObservation):
    """Provide an interface to single-dish observation.

    Parameters
    ----------
    pointings: Directions
        Contains all information on the observing directions.
    data: numpy.ndarray
        Contains the measured intensities. Shape (n_polarizations, n_rows,
        n_channels).
    weight : numpy.ndarray
        Contains the information from the WEIGHT or SPECTRUM_WEIGHT column.
        This is in many cases the inverse of the thermal noise covariance.
        Shape same as `data`.
    polarization : Polarization
        Polarization information of the data set.
    freq : numpy.ndarray
        Contains the measured frequencies. Shape (n_channels)
    """

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

        self._eq_attributes = "_polarization", "_freq", "_pointings", "_vis", "_weight"

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


class Observation(BaseObservation):
    """Provide an interface to an interferometric observation.

    This class contains all the data and information about an observation.
    It supports a single field (phase center) and a single spectral window.

    Parameters
    ----------
    antenna_positions : AntennaPositions
        Contains all information on antennas and baselines.
    vis : numpy.ndarray
        Contains the measured visibilities. Shape (n_polarizations, n_rows, n_channels)
    weight : numpy.ndarray
        Contains the information from the WEIGHT or SPECTRUM_WEIGHT column.
        This is in many cases the inverse of the thermal noise covariance. Shape same as vis.
    polarization : Polarization
        Polarization information of the data set.
    freq : numpy.ndarray
        Contains the measured frequencies. Shape (n_channels)
    direction : Direction
        Direction information of the data set.

    Note
    ----
    vis and weight must have the same shape.
    """

    def __init__(self, antenna_positions, vis, weight, polarization, freq, direction):
        nrows = len(antenna_positions)
        my_assert_isinstance(direction, Direction)
        my_assert_isinstance(polarization, Polarization)
        my_assert_isinstance(antenna_positions, AntennaPositions)
        my_asserteq(weight.shape, vis.shape)
        my_asserteq(vis.shape, (len(polarization), nrows, len(freq)))
        my_asserteq(nrows, vis.shape[1])
        if vis.dtype != np.complex64:
            print(f"Warning: vis.dtype is {vis.dtype}. Casting to np.complex64")
            vis = vis.astype(np.complex64)
        if weight.dtype != np.float32:
            print(f"Warning: weight.dtype is {weight.dtype}. Casting to np.float32")
            weight = weight.astype(np.float32)
        my_asserteq(vis.dtype, np.complex64)
        my_asserteq(weight.dtype, np.float32)
        my_assert(np.all(weight >= 0.0))
        my_assert(np.all(np.isfinite(vis[weight > 0.])))
        my_assert(np.all(np.isfinite(weight)))

        vis.flags.writeable = False
        weight.flags.writeable = False

        self._antpos = antenna_positions
        self._vis = vis
        self._weight = weight
        self._polarization = polarization
        self._freq = freq
        self._direction = direction

        self._eq_attributes = "_direction", "_polarization", "_freq", "_antpos", "_vis", "_weight"

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
        dct = np.load(file_name)
        antpos = []
        for ii in range(4):
            val = dct[f"antpos{ii}"]
            if val.size == 0:
                val = None
            antpos.append(val)
        antpos = AntennaPositions.from_list(antpos)
        pol = Polarization.from_list(dct["polarization"])
        direction = Direction.from_list(dct["direction"])
        vis = dct["vis"]
        wgt = dct["weight"]
        freq = dct["freq"]
        if lo_hi_index is not None:
            slc = slice(*lo_hi_index)
            # Convert view into its own array
            vis = vis[..., slc].copy()
            wgt = wgt[..., slc].copy()
            freq = freq[slc].copy()
        del dct
        return Observation(antpos, vis, wgt, pol, freq, direction)

    def flags_to_nan(self):
        if self.fraction_useful == 1.:
            return self
        vis = self._vis.copy()
        vis[self.flags.val] = np.nan
        return Observation(self._antpos, vis, self._weight, self._polarization, self._freq,
                           self._direction)

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
            lo, hi = ift.utilities.shareRange( n_imaging_bands, comm.Get_size(), comm.Get_rank())
            local_imaging_bands = range(lo, hi)
        full_obs = Observation.load(file_name)
        obs_list = []
        for ii in local_imaging_bands:
            slc = slice(*ift.utilities.shareRange(len(global_freqs), n_imaging_bands, ii))
            obs_list.append(full_obs.get_freqs_by_slice(slc))
        nu0 = global_freqs.mean()
        return obs_list, nu0

    def __getitem__(self, slc, copy=False):
        ap = self._antpos[slc]
        vis = self._vis[slc]
        wgt = self._weight[:, slc]
        if copy:
            ap = ap.copy()
            vis = vis.copy()
            wgt = wgt.copy()
        return Observation(ap, vis, wgt, self._polarization, self._freq, self._direction)

    def get_freqs(self, frequency_list, copy=False):
        """Return observation that contains a subset of the present frequencies

        Parameters
        ----------
        frequency_list : list
            List of indices that shall be returned
        """
        mask = np.zeros(self.nfreq, dtype=bool)
        mask[frequency_list] = 1
        return self.get_freqs_by_slice(mask, copy)

    def get_freqs_by_slice(self, slc, copy=False):
        vis = self._vis[..., slc]
        wgt = self._weight[..., slc]
        freq = self._freq[slc]
        if copy:
            vis = vis.copy()
            wgt = wgt.copy()
            freq = freq.copy()
        return Observation( self._antpos, vis, wgt, self._polarization, freq, self._direction)

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

    def flag_baseline(self, ant1_index, ant2_index):
        ant1 = self.antenna_positions.ant1
        ant2 = self.antenna_positions.ant2
        if ant1 is None or ant2 is None:
            raise RuntimeError("The calibration information needed for flagging a baseline is not "
                               "available. Please import the measurement set with "
                               "`with_calib_info=True`.")
        assert np.all(ant1 < ant2)
        ind = np.logical_and(ant1 == ant1_index, ant2 == ant2_index)
        wgt = self._weight.copy()
        wgt[:, ind] = 0.
        print("INFO: Flag baseline {ant1_index}-{ant2_index}, {np.sum(ind)}/{obs.nrows} rows flagged.")
        return Observation(self._antpos, self._vis, wgt, self._polarization, self._freq, self._direction)

    @property
    def uvw(self):
        return self._antpos.uvw

    @property
    def antenna_positions(self):
        return self._antpos

    def effective_uvw(self):
        out = np.einsum("ij,k->jik", self.uvw, self._freq / SPEEDOFLIGHT)
        my_asserteq(out.shape, (3, self.nrow, self.nfreq))
        return out

    def effective_uvwlen(self):
        arr = np.outer(self.uvwlen(), self._freq / SPEEDOFLIGHT)
        arr = np.broadcast_to(arr[None], self._dom.shape)
        return ift.makeField(self._dom, arr)

    def uvwlen(self):
        return np.linalg.norm(self.uvw, axis=1)


def tmin_tmax(*args):
    """Compute beginning and end time of list of observations.

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
    """Compute set of antennas of list of observations

    Parameters
    ----------
    args : Observation or list of Observation

    Returns
    -------
    set
        Set of antennas
    """
    my_assert_isinstance(*args, Observation)
    antennas = set()
    for oo in args:
        antennas = antennas | oo.antenna_positions.unique_antennas()
    return antennas


def unique_times(*args):
    """Compute set of time stamps of list of observations

    Parameters
    ----------
    args : Observation or list of Observation

    Returns
    -------
    set
        Set of time stamps
    """
    my_assert_isinstance(*args, Observation)
    times = set()
    for oo in args:
        times = times | oo.antenna_positions.unique_times()
    return times
