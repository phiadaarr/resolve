# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society

from os.path import isdir, join, splitext

import numpy as np

from .antenna_positions import AntennaPositions
from .direction import Direction
from .observation import Observation
from .polarization import Polarization
from .util import my_asserteq, my_assert


CFG = {'readonly': True, 'ack': False}


def ms2observations(ms, data_column, with_calib_info, spectral_window=None, polarizations="all"):
    # FIXME Support to read in only selected channels
    """

    If WEIGHT_SPECTRUM is available this column is used for weighting.
    Otherwise fall back to WEIGHT.

    If "stokesi" is chosen then visibilities where one of the two polarizations
    is flagged are flagged.

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
        Index of spectral window which shall be imported. If only one spectral
        window is present, defaults to important it. If several spectral
        windows are present, this parameter is non-optional.
    polarizations
        "all":     All polarizations are imported.
        "stokesi": Only LL/RR or XX/YY polarizations are imported and averaged
                   on the fly.
        List of strings: Strings can be "XX", "YY", "XY", "YX", "LL", "LR",
                   "RL", "RR". The respective polarizations are loaded.

    Note
    ----
    We cannot import multiple spectral windows into one Observation instance
    because it is not guaranteed by the measurement set data structure that all
    baselines are present in all spectral windows.
    """
    from casacore.tables import table

    if not isdir(ms) or splitext(ms)[1] != '.ms':
        raise RuntimeError
    if isinstance(polarizations, str):
        polarizations = [polarizations]
    if ("stokesi" in polarizations or "all" in polarizations) and len(polarizations) > 1:
        raise ValueError

    # Spectral windows
    my_assert(spectral_window is None or (isinstance(spectral_window, int) and spectral_window >= 0))
    with table(join(ms, 'SPECTRAL_WINDOW'), **CFG) as t:
        freq = t.getcol('CHAN_FREQ')
        nspws = freq.shape[0]
        if spectral_window is None and nspws > 1:
            raise RuntimeError("Need to specify spectral window.")
        spectral_window = 0 if nspws == 1 else int(spectral_window)
        freq = freq[spectral_window]

    # Polarization
    with table(join(ms, 'POLARIZATION'), **CFG) as t:
        pol = t.getcol('CORR_TYPE')
        my_asserteq(pol.ndim, 2)
        my_asserteq(pol.shape[0], 1)
        pol = list(pol[0])  # Not clear what the first dimension is used for
        polobj = Polarization(pol)
        if polarizations[0] == "stokesi":
            pol_ind = polobj.stokes_i_indices()
            polobj = polobj.restrict_to_stokes_i()
            pol_summation = True
        elif polarizations[0] != "all":
            pol_ind = [pol.index(ii) for ii in polarizations]
            polobj = polobj.restrict_by_name(polarizations)
            pol_summation = False
        elif polarizations[0] == "all":
            pol_ind = None
            pol_summation = False
        else:
            raise RuntimeError

    # Field
    with table(join(ms, 'FIELD'), **CFG) as t:
        equinox = t.coldesc('REFERENCE_DIR')['desc']['keywords']['MEASINFO']['Ref']
        equinox = str(equinox)[1:]
        # TODO Put proper support for equinox here
        if equinox == "1950_VLA":
            equinox = 1950
        dirs = []
        for pc in t.getcol('REFERENCE_DIR'):
            my_asserteq(pc.shape, (1, 2))
            dirs.append(Direction(pc[0], equinox))
        dirs = tuple(dirs)

    # TODO Determine which observation is calibration observation
    # TODO Import name of source
    observations = []
    for ifield, direction in enumerate(dirs):
        uvw, ant1, ant2, time, freq_out, vis, wgt, flags = read_ms_i(
            ms, data_column, freq, ifield, spectral_window, pol_ind,
            pol_summation, with_calib_info)
        if np.any(flags):
            if not wgt.flags.writeable:
                # This is necessary because otherwise we can store the weight
                # array with less memory (created with np.broadcast_to)
                wgt = wgt.copy()
            wgt[flags] = 0.
        del flags
        vis[wgt == 0] = 0.
        vis = _ms2resolve_transpose(vis)
        wgt = _ms2resolve_transpose(wgt)
        antpos = AntennaPositions(uvw, ant1, ant2, time)
        observations.append(Observation(antpos, vis, wgt, polobj, freq_out,
                                        direction))
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


def read_ms_i(name, data_column, freq, field, spectral_window, pol_indices, pol_summation, with_calib_info):
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

    with table(name, readonly=True, ack=False) as t:
        fullwgt, weightcol = _determine_weighting(t)
        nrow = t.nrows()
        active_rows = np.ones(nrow, dtype=np.bool)
        active_channels = np.zeros(nchan, dtype=np.bool)
        step = max(1, nrow//100)  # how many rows to read in every step

        # Determine which subset of rows/channels we need to input
        start = 0
        while start < nrow:
            stop = min(nrow, start+step)
            tflags = t.getcol('FLAG', startrow=start, nrow=stop-start)[..., pol_indices]
            twgt = t.getcol(weightcol, startrow=start, nrow=stop-start)[..., pol_indices]
            # ncorr = tflags.shape[2]
            if pol_summation:
                tflags = np.any(tflags.astype(np.bool), axis=-1)
                twgt = np.sum(twgt, axis=-1)
            if twgt.ndim == 3:  # with WEIGHT_SPECTRUM
                tflags[twgt == 0] = True

            # Select field and spectral window
            tfieldid = t.getcol("FIELD_ID", startrow=start, nrow=stop-start)
            tflags[tfieldid != field] = True
            tspw = t.getcol("FIELD_ID", startrow=start, nrow=stop-start)
            tflags[tspw != spectral_window] = True

            npol = tflags.shape[2]
            if not pol_summation:
                tflags = np.all(tflags, axis=2)
            assert tflags.ndim == 2
            active_rows[start:stop] = np.invert(np.all(tflags, axis=-1))
            active_channels = np.logical_or(active_channels, np.invert(np.all(tflags, axis=0)))
            start = stop
        nrealrows, nrealchan = np.sum(active_rows), np.sum(active_channels)
        if nrealrows == 0 or nrealchan == 0:
            raise RuntimeError('Empty data set')

        # Read in data
        start, realstart = 0, 0
        shp = (nrealrows, nrealchan)
        wgtshp = (nrealrows, nrealchan) if fullwgt else (nrealrows,)
        if not pol_summation:
            npol = npol if pol_indices == slice(None) else len(pol_indices)
            shp = shp + (npol,)
            wgtshp = wgtshp + (npol,)
        vis = np.empty(shp, dtype=np.complex64)
        flags = np.empty(shp, dtype=np.bool)
        wgt = np.empty(wgtshp, dtype=np.float32)
        while start < nrow:
            stop = min(nrow, start+step)
            realstop = realstart+np.sum(active_rows[start:stop])
            if realstop > realstart:
                allrows = stop-start == realstop-realstart
                tvis = t.getcol(data_column, startrow=start, nrow=stop-start)[..., pol_indices]
                assert tvis.dtype == np.complex64
                if pol_summation:
                    # FIXME Implement weighted sum here
                    tvis = np.sum(tvis, axis=-1)
                if not allrows:
                    tvis = tvis[active_rows[start:stop]]
                tvis = tvis[:, active_channels]
                tflags = t.getcol("FLAG", startrow=start, nrow=stop-start)[..., pol_indices]
                if pol_summation:
                    tflags = np.any(tflags.astype(np.bool), axis=-1)
                if not allrows:
                    tflags = tflags[active_rows[start:stop]]
                tflags = tflags[:, active_channels]
                twgt = t.getcol(weightcol, startrow=start, nrow=stop-start)[..., pol_indices]
                assert twgt.dtype == np.float32
                if pol_summation:
                    twgt = np.sum(twgt, axis=-1)
                if not allrows:
                    twgt = twgt[active_rows[start:stop]]
                if fullwgt:
                    twgt = twgt[:, active_channels]
#                tflags[twgt==0] = True

                vis[realstart:realstop] = tvis
                wgt[realstart:realstop] = twgt
                flags[realstart:realstop] = tflags

            start, realstart = stop, realstop
        uvw = t.getcol("UVW")[active_rows]
        if bool(with_calib_info):
            ant1 = np.ascontiguousarray(t.getcol("ANTENNA1")[active_rows])
            ant2 = np.ascontiguousarray(t.getcol("ANTENNA2")[active_rows])
            time = np.ascontiguousarray(t.getcol("TIME")[active_rows])
        else:
            ant1 = ant2 = time = None

    print('# Rows: {} ({} fully flagged or not selected)'.format(nrow, nrow-vis.shape[0]))
    print('# Channels: {} ({} fully flagged)'.format(nchan, nchan-vis.shape[1]))
    print('# Correlations: {}'.format(1 if pol_summation else vis.shape[2]))
    print('Full weights' if fullwgt else 'Row-only weights')
    nflagged = np.sum(flags) + (nrow-nrealrows)*nchan + (nchan-nrealchan)*nrow
    print("{} % flagged".format(nflagged/(nrow*nchan)*100))
    freq = freq[active_channels]

    # blow up wgt to the right dimensions if necessary
    if not fullwgt:
        wgt = np.broadcast_to(wgt[:, None], vis.shape)
    my_asserteq(wgt.shape, vis.shape)
    return (np.ascontiguousarray(uvw),
            ant1, ant2, time,
            np.ascontiguousarray(freq),
            np.ascontiguousarray(vis),
            np.ascontiguousarray(wgt),# if fullwgt else wgt,
            flags)
