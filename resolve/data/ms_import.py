# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society

import os
from os.path import isdir, join, splitext

import numpy as np

from ..util import my_assert, my_assert_isinstance, my_asserteq
from .antenna_positions import AntennaPositions
from .auxiliary_table import AuxiliaryTable
from .direction import Direction
from .observation import Observation
from .polarization import Polarization


def ms_table(path):
    from casacore.tables import table
    return table(path, readonly=True, ack=False)


def ms2observations(ms, data_column, with_calib_info, spectral_window,
                    polarizations="all", channel_slice=slice(None),
                    ignore_flags=False):
    """Read and convert a given measurement set into an array of :class:`Observation`

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
        "stokesiavg": Only LL/RR or XX/YY polarizations are imported and averaged on the fly.
        List of strings: Strings can be "XX", "YY", "XY", "YX", "LL", "LR", "RL", "RR". The respective polarizations are loaded.
    channel_slice : slice
        Slice of selected channels. Default: select all channels
        FIXME Select channels by indices
    ignore_flags : bool
        If True, the whole measurement set is imported irrespective of the
        flags. Default is false.

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
    if ms[-1] == "/":
        ms = ms[:-1]
    if not isdir(ms):
        raise RuntimeError
    if ms == ".":
        ms = os.getcwd()
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

    # Polarization
    with ms_table(join(ms, "POLARIZATION")) as t:
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
    with ms_table(join(ms, "FIELD")) as t:
        # FIXME Put proper support for equinox here
        # FIXME Eventually get rid of this and use auxiliary table
        try:
            equinox = t.coldesc("REFERENCE_DIR")["desc"]["keywords"]["MEASINFO"]["Ref"]
            equinox = str(equinox)[1:]
            if equinox == "1950_VLA":
                equinox = 1950
        except KeyError:
            equinox = 2000
        dirs = []
        for pc in t.getcol("REFERENCE_DIR"):
            my_asserteq(pc.shape, (1, 2))
            dirs.append(Direction(pc[0], equinox))
        dirs = tuple(dirs)

    auxtables = {}
    with ms_table(join(ms, "ANTENNA")) as t:
        keys = ["NAME", "STATION", "TYPE", "MOUNT", "POSITION", "OFFSET", "DISH_DIAMETER"]
        auxtables["ANTENNA"] = AuxiliaryTable({kk: t.getcol(kk) for kk in keys})
    with ms_table(join(ms, "SPECTRAL_WINDOW")) as t:
        keys = ["NAME", "REF_FREQUENCY", "CHAN_FREQ", "CHAN_WIDTH", "MEAS_FREQ_REF", "EFFECTIVE_BW",
                "RESOLUTION", "TOTAL_BANDWIDTH", "NET_SIDEBAND", "IF_CONV_CHAIN", "FREQ_GROUP",
                "FREQ_GROUP_NAME"]
        dct = {kk: t.getcol(kk, startrow=spectral_window, nrow=1) for kk in keys}
        auxtables["SPECTRAL_WINDOW"] = AuxiliaryTable(dct)

    # FIXME Determine which observation is calibration observation
    observations = []
    for ifield, direction in enumerate(dirs):
        with ms_table(join(ms, "FIELD")) as t:
            keys = ["NAME", "CODE", "TIME", "NUM_POLY", "DELAY_DIR", "PHASE_DIR", "REFERENCE_DIR",
                    "SOURCE_ID"]
            dct = {kk: t.getcol(kk, startrow=ifield, nrow=1) for kk in keys}
            auxtables["FIELD"] = AuxiliaryTable(dct)

        mm = read_ms_i(ms, data_column, ifield, spectral_window, pol_ind, pol_summation,
                       with_calib_info, channel_slice, ignore_flags,)
        if mm is None:
            print(f"{ms}, field #{ifield} is empty or fully flagged")
            observations.append(None)
            continue
        if mm["ptg"] is not None:
            raise NotImplementedError
        antpos = AntennaPositions(mm["uvw"], mm["ant1"], mm["ant2"], mm["time"])
        obs = Observation(antpos, mm["vis"], mm["wgt"], polobj, mm["freq"], direction,
                          auxiliary_tables=auxtables)
        observations.append(obs)
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


def read_ms_i(name, data_column, field, spectral_window, pol_indices, pol_summation,
              with_calib_info, channel_slice, ignore_flags):

    # Freq
    with ms_table(join(name, "SPECTRAL_WINDOW")) as t:
        freq = t.getcol("CHAN_FREQ", startrow=spectral_window, nrow=1)
    my_asserteq(freq.ndim, 2)
    my_asserteq(freq.shape[0], 1)
    freq = freq[0]
    my_assert(len(freq) > 0)
    nchan = len(freq)

    assert pol_indices is None or isinstance(pol_indices, list)
    if pol_indices is None:
        pol_indices = slice(None)
    if pol_summation:
        my_asserteq(len(pol_indices), 2)

    with ms_table(name) as t:
        # FIXME Get rid of fullwgt
        fullwgt, weightcol = _determine_weighting(t)
        nrow = t.nrows()
        nmspol = t.getcol("FLAG", startrow=0, nrow=1).shape[2]
        print("Measurement set visibilities:")
        print(f"  shape: ({nrow}, {nchan}, {nmspol})")
        active_rows = np.ones(nrow, dtype=np.bool)
        active_channels = np.zeros(nchan, dtype=np.bool)
        step = max(1, nrow // 100)  # how many rows to read in every step

        # Check if data column is available
        t.getcol(data_column, startrow=0, nrow=10)

        # Determine which subset of rows/channels we need to input
        start = 0
        while start < nrow:
            print("First pass:", f"{(start/nrow*100):.1f}%", end="\r")
            stop = min(nrow, start + step)
            tflags = _conditional_flags(t, start, stop, pol_indices, ignore_flags)
            twgt = t.getcol(weightcol, startrow=start, nrow=stop - start)[..., pol_indices]
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
            return None

    # Freq
    freq = freq[active_channels]

    # Vis, wgt, (flags)
    with ms_table(name) as t:
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

                # Weights
                twgt = t.getcol(weightcol, startrow=start, nrow=stop - start)[..., pol_indices]
                assert twgt.dtype == np.float32
                if not fullwgt:
                    twgt = np.repeat(twgt[:, None], nchan, axis=1)
                if not allrows:
                    twgt = twgt[active_rows[start:stop]]
                twgt = twgt[:, active_channels]

                # Vis
                tvis = t.getcol(data_column, startrow=start, nrow=stop - start)[..., pol_indices]
                assert tvis.dtype == np.complex64
                if not allrows:
                    tvis = tvis[active_rows[start:stop]]
                tvis = tvis[:, active_channels]

                # Flags
                tflags = _conditional_flags(t, start, stop, pol_indices, ignore_flags)
                if not allrows:
                    tflags = tflags[active_rows[start:stop]]
                tflags = tflags[:, active_channels]

                # Polarization summation
                assert twgt.ndim == tflags.ndim == 3
                assert tflags.dtype == np.bool
                if not ignore_flags:
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
    print("Selected:", 10 * " ")
    print(f"  shape: {vis.shape}")
    print(f"  flagged: {(1.0-np.sum(wgt!=0)/wgt.size)*100:.1f} %")

    # UVW
    with ms_table(name) as t:
        uvw = np.ascontiguousarray(t.getcol("UVW")[active_rows])

    # Calibration info
    if with_calib_info:
        with ms_table(name) as t:
            ant1 = np.ascontiguousarray(t.getcol("ANTENNA1")[active_rows])
            ant2 = np.ascontiguousarray(t.getcol("ANTENNA2")[active_rows])
            time = np.ascontiguousarray(t.getcol("TIME")[active_rows])
    else:
        ant1 = ant2 = time = None

    # Pointing
    with ms_table(join(name, "POINTING")) as t:
        if t.nrows() == 0:
            ptg = None
        else:
            ptg = np.empty((nrealrows, 1, 2), dtype=np.float64)
            start, realstart = 0, 0
            while start < nrow:
                print("Second pass:", f"{(start/nrow*100):.1f}%", end="\r")
                stop = min(nrow, start + step)
                realstop = realstart + np.sum(active_rows[start:stop])
                if realstop > realstart:
                    allrows = stop - start == realstop - realstart
                    tptg = t.getcol("DIRECTION", startrow=start, nrow=stop - start)
                    tptg = tptg[active_rows[start:stop]]
                    ptg[realstart:realstop] = tptg
                start, realstart = stop, realstop

    my_asserteq(wgt.shape, vis.shape)
    vis = np.ascontiguousarray(_ms2resolve_transpose(vis))
    wgt = np.ascontiguousarray(_ms2resolve_transpose(wgt))
    vis[wgt == 0] = 0.0
    return {"uvw": uvw, "ant1": ant1, "ant2": ant2, "time": time, "freq": freq, "vis": vis, "wgt": wgt, "ptg": ptg}


def ms_n_spectral_windows(ms):
    with ms_table(join(ms, "SPECTRAL_WINDOW")) as t:
        n_spectral_windows = t.nrows()
    return n_spectral_windows


def _conditional_flags(table, start, stop, pol_indices, ignore):
    tflags = table.getcol("FLAG", startrow=start, nrow=stop - start)[..., pol_indices]
    if ignore:
        tflags = np.zeros_like(tflags)
    return tflags
