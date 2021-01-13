# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society

from os.path import isdir, join, splitext

import numpy as np

from .antenna_positions import AntennaPositions
from .direction import Direction
from .observation import Observation
from .polarization import Polarization
from .util import my_assert, my_asserteq, my_assert_isinstance

_CASACORE_TABLE_CFG = {"readonly": True, "ack": False}


def ms2observations(
    ms,
    data_column,
    with_calib_info,
    spectral_window,
    polarizations="all",
    channel_slice=slice(None),
):
    """
    Reads and converts a given measurement set into an array of :class:`Observation`

    If WEIGHT_SPECTRUM is available this column is used for weighting.
    Otherwise fall back to WEIGHT.

    Parameters
    ----------
    ms : string
        Folder name of measurement set
    data_column : string
        Column of measurement set from which the visibilities are read.
        Typically either "DATA" or "CORRECTED_DATA".
    with_calib_info : bool
        Reads in all information necessary for calibration, if True. If only
        imaging shall be performed, this can be set to False.
    spectral_window : int
        Index of spectral window which shall be imported.
    polarizations
        "all":     All polarizations are imported.
        "stokesi": Only LL/RR or XX/YY polarizations are imported.
        "stokesiavg": Only LL/RR or XX/YY polarizations are imported and averaged
                   on the fly.
        List of strings: Strings can be "XX", "YY", "XY", "YX", "LL", "LR",
                   "RL", "RR". The respective polarizations are loaded.
    channel_slice : slice
        Slice of selected channels. Default: select all channels
        FIXME Select channels by indices

    Returns
    -------
    array[Observation]
        an array of :class:`Observation` found in the measurement set

    Note
    ----
    We cannot import multiple spectral windows into one Observation instance
    because it is not guaranteed by the measurement set data structure that all
    baselines are present in all spectral windows.
    """
    from casacore.tables import table

    if ms[-1] == "/":
        ms = ms[:-1]
    if not isdir(ms) or splitext(ms)[1].lower() != ".ms":
        raise RuntimeError
    if isinstance(polarizations, str):
        polarizations = [polarizations]
    if (
        "stokesiavg" in polarizations
        or "stokesi" in polarizations
        or "all" in polarizations
    ) and len(polarizations) > 1:
        raise ValueError

    # Spectral windows
    my_assert_isinstance(channel_slice, slice)
    my_assert_isinstance(spectral_window, int)
    my_assert(spectral_window >= 0)
    my_assert(spectral_window < ms_n_spectral_windows(ms))
    with table(join(ms, "SPECTRAL_WINDOW"), **_CASACORE_TABLE_CFG) as t:
        freq = t.getcol("CHAN_FREQ")[spectral_window]

    # Polarization
    with table(join(ms, "POLARIZATION"), **_CASACORE_TABLE_CFG) as t:
        pol = t.getcol("CORR_TYPE")
        my_asserteq(pol.ndim, 2)
        my_asserteq(pol.shape[0], 1)
        pol = list(pol[0])  # Not clear what the first dimension is used for
        polobj = Polarization(pol)
        if polarizations[0] == "stokesi":
            polarizations = ["LL", "RR"] if polobj.circular() else ["XX", "YY"]
        if polarizations[0] == "stokesiavg":
            pol_ind = polobj.stokes_i_indices()
            polobj = Polarization.trivial()
            pol_summation = True
        elif polarizations[0] == "all":
            pol_ind = None
            pol_summation = False
        else:
            polobj = polobj.restrict_by_name(polarizations)
            pol_ind = [polobj.to_str_list().index(ii) for ii in polarizations]
            pol_summation = False

    # Field
    with table(join(ms, "FIELD"), **_CASACORE_TABLE_CFG) as t:
        equinox = t.coldesc("REFERENCE_DIR")["desc"]["keywords"]["MEASINFO"]["Ref"]
        equinox = str(equinox)[1:]
        # FIXME Put proper support for equinox here
        if equinox == "1950_VLA":
            equinox = 1950
        dirs = []
        for pc in t.getcol("REFERENCE_DIR"):
            my_asserteq(pc.shape, (1, 2))
            dirs.append(Direction(pc[0], equinox))
        dirs = tuple(dirs)

    # FIXME Determine which observation is calibration observation
    # FIXME Import name of source
    observations = []
    for ifield, direction in enumerate(dirs):
        uvw, ant1, ant2, time, freq_out, vis, wgt = read_ms_i(
            ms,
            data_column,
            freq,
            ifield,
            spectral_window,
            pol_ind,
            pol_summation,
            with_calib_info,
            channel_slice,
        )
        vis[wgt == 0] = 0.0
        vis = _ms2resolve_transpose(vis)
        wgt = _ms2resolve_transpose(wgt)
        antpos = AntennaPositions(uvw, ant1, ant2, time)
        observations.append(Observation(antpos, vis, wgt, polobj, freq_out, direction))
    return observations


def _ms2resolve_transpose(arr):
    my_asserteq(arr.ndim, 3)
    return np.ascontiguousarray(np.transpose(arr, (2, 0, 1)))


def _determine_weighting(t):
    fullwgt = False
    weightcol = "WEIGHT"
    try:
        t.getcol("WEIGHT_SPECTRUM", startrow=0, nrow=1)
        weightcol = "WEIGHT_SPECTRUM"
        fullwgt = True
    except RuntimeError:
        pass
    return fullwgt, weightcol


def read_ms_i(
    name,
    data_column,
    freq,
    field,
    spectral_window,
    pol_indices,
    pol_summation,
    with_calib_info,
    channel_slice,
):
    from casacore.tables import table

    freq = np.array(freq)
    my_asserteq(freq.ndim, 1)
    my_assert(len(freq) > 0)
    nchan = len(freq)
    assert pol_indices is None or isinstance(pol_indices, list)
    if pol_indices is None:
        pol_indices = slice(None)
    if pol_summation:
        my_asserteq(len(pol_indices), 2)

    with table(name, **_CASACORE_TABLE_CFG) as t:
        # FIXME Get rid of fullwgt
        fullwgt, weightcol = _determine_weighting(t)
        nrow = t.nrows()
        nmspol = t.getcol("FLAG", startrow=0, nrow=1).shape[2]
        print("Measurement set visibilities:")
        print(f"  shape: ({nrow}, {nchan}, {nmspol})")
        active_rows = np.ones(nrow, dtype=np.bool)
        active_channels = np.zeros(nchan, dtype=np.bool)
        step = max(1, nrow // 100)  # how many rows to read in every step

        # Determine which subset of rows/channels we need to input
        start = 0
        while start < nrow:
            print("First pass:", f"{(start/nrow*100):.1f}%", end="\r")
            stop = min(nrow, start + step)
            tflags = t.getcol("FLAG", startrow=start, nrow=stop - start)[
                ..., pol_indices
            ]
            twgt = t.getcol(weightcol, startrow=start, nrow=stop - start)[
                ..., pol_indices
            ]
            if channel_slice != slice(None):
                tchslcflags = np.ones_like(tflags)
                tchslcflags[:, channel_slice] = False
                tflags = np.logical_or(tflags, tchslcflags)
            if not fullwgt:
                twgt = np.repeat(twgt[:, None], nchan, axis=1)
            my_asserteq(twgt.ndim, 3)
            npol = tflags.shape[2]
            if pol_summation:
                tflags = np.any(tflags.astype(np.bool), axis=2)[..., None]
                twgt = np.sum(twgt, axis=2)[..., None]
            tflags[twgt == 0] = True

            # Select field and spectral window
            tfieldid = t.getcol("FIELD_ID", startrow=start, nrow=stop - start)
            tflags[tfieldid != field] = True
            tspw = t.getcol("DATA_DESC_ID", startrow=start, nrow=stop - start)
            tflags[tspw != spectral_window] = True

            # Inactive if all polarizations are flagged
            assert tflags.ndim == 3
            tflags = np.all(tflags, axis=2)
            active_rows[start:stop] = np.invert(np.all(tflags, axis=1))
            active_channels = np.logical_or(
                active_channels, np.invert(np.all(tflags, axis=0))
            )
            start = stop
        nrealrows, nrealchan = np.sum(active_rows), np.sum(active_channels)
        if nrealrows == 0 or nrealchan == 0:
            raise RuntimeError("Empty data set")

        # Create output arrays
        if pol_summation:
            npol = 1
        else:
            if pol_indices != slice(None):
                npol = len(pol_indices)
            else:
                npol = npol
        shp = (nrealrows, nrealchan, npol)
        vis = np.empty(shp, dtype=np.complex64)
        wgt = np.empty(shp, dtype=np.float32)

        # Read in data
        start, realstart = 0, 0
        while start < nrow:
            print("Second pass:", f"{(start/nrow*100):.1f}%", end="\r")
            stop = min(nrow, start + step)
            realstop = realstart + np.sum(active_rows[start:stop])
            if realstop > realstart:
                allrows = stop - start == realstop - realstart

                twgt = t.getcol(weightcol, startrow=start, nrow=stop - start)[
                    ..., pol_indices
                ]
                assert twgt.dtype == np.float32
                if not fullwgt:
                    twgt = np.repeat(twgt[:, None], nchan, axis=1)
                if not allrows:
                    twgt = twgt[active_rows[start:stop]]
                twgt = twgt[:, active_channels]

                tvis = t.getcol(data_column, startrow=start, nrow=stop - start)[
                    ..., pol_indices
                ]
                assert tvis.dtype == np.complex64
                if not allrows:
                    tvis = tvis[active_rows[start:stop]]
                tvis = tvis[:, active_channels]

                tflags = t.getcol("FLAG", startrow=start, nrow=stop - start)[
                    ..., pol_indices
                ]
                if not allrows:
                    tflags = tflags[active_rows[start:stop]]
                tflags = tflags[:, active_channels]

                assert twgt.ndim == tflags.ndim == 3
                assert tflags.dtype == np.bool
                twgt = twgt * (~tflags)
                if pol_summation:
                    assert twgt.shape[2] == 2
                    # Noise-weighted average
                    tvis = np.sum(twgt * tvis, axis=-1)[..., None]
                    twgt = np.sum(twgt, axis=-1)[..., None]
                    tvis /= twgt
                del tflags
                vis[realstart:realstop] = tvis
                wgt[realstart:realstop] = twgt

            start, realstart = stop, realstop
        uvw = t.getcol("UVW")[active_rows]
        if with_calib_info:
            ant1 = np.ascontiguousarray(t.getcol("ANTENNA1")[active_rows])
            ant2 = np.ascontiguousarray(t.getcol("ANTENNA2")[active_rows])
            time = np.ascontiguousarray(t.getcol("TIME")[active_rows])
        else:
            ant1 = ant2 = time = None
    print("Selected:", 10 * " ")
    print(f"  shape: {vis.shape}")
    print(f"  flagged: {(1.0-np.sum(wgt!=0)/wgt.size)*100:.1f} %")
    freq = freq[active_channels]

    my_asserteq(wgt.shape, vis.shape)
    return (
        np.ascontiguousarray(uvw),
        ant1,
        ant2,
        time,
        np.ascontiguousarray(freq),
        np.ascontiguousarray(vis),
        np.ascontiguousarray(wgt),
    )


def ms_n_spectral_windows(ms):
    from casacore.tables import table

    with table(join(ms, "SPECTRAL_WINDOW"), **_CASACORE_TABLE_CFG) as t:
        freq = t.getcol("CHAN_FREQ")
    return freq.shape[0]
