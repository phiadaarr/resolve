# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

from os.path import isdir, join, splitext

import numpy as np

from .antenna_positions import AntennaPositions
from .direction import Direction
from .observation import Observation
from .polarization import Polarization
from .util import my_asserteq


def ms2observations(ms, data_column, spectral_window=None):
    from casacore.tables import table
    CFG = {'readonly': True, 'ack': False}
    if not isdir(ms) or splitext(ms)[1] != '.ms':
        raise RuntimeError

    print('Load SPECTRAL_WINDOW table')
    with table(join(ms, 'SPECTRAL_WINDOW'), **CFG) as t:
        freqs = t.getcol('CHAN_FREQ')
    print('Load POLARIZATION table')
    with table(join(ms, 'POLARIZATION'), **CFG) as t:
        polarization = t.getcol('CORR_TYPE')
        my_asserteq(polarization.ndim, 2)
        my_asserteq(polarization.shape[0], 1)
        polarization = Polarization(polarization[0])
    print('Load FIELD table')
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
    print('Load main table')
    with table(ms, **CFG) as t:
        vis = t.getcol(data_column)
        print(f'vis data type is {vis.dtype}')
        uvw = t.getcol('UVW')
        if 'WEIGHT_SPECTRUM' in t.colnames():
            try:
                weight = t.getcol('WEIGHT_SPECTRUM')
            except RuntimeError:
                print('Invalid WEIGHT_SPECTRUM column. Use less informative WEIGHT column instead.')
                weight = t.getcol('WEIGHT')
        else:
            print('No WEIGHT_SPECTRUM column. Use less informative WEIGHT column instead.')
            weight = t.getcol('WEIGHT')
        flags = t.getcol('FLAG')
        fieldid = t.getcol('FIELD_ID')
        spw = t.getcol("DATA_DESC_ID")
        ant1 = t.getcol("ANTENNA1")
        ant2 = t.getcol("ANTENNA2")
        time = t.getcol("TIME")
        antpos = AntennaPositions(uvw, ant1, ant2, time)

    nspws = len(np.unique(spw))
    if nspws > 1:
        if spectral_window is None:
            raise RuntimeError
        spwmask = spw == spectral_window
        freqs = freqs[spectral_window]
    else:
        freqs = freqs[0]

    if weight.ndim == 2:
        # TODO Memory: Do not repeat this array and tell it the likelihood instead
        weight = np.repeat(weight[:, None], vis.shape[1], axis=1)
    my_asserteq(weight.shape, flags.shape, vis.shape)
    # Convention: can use flag as index array: vis[flags] gives out good visibilities
    flags = ~flags
    inds = weight == 0
    weight[inds] = 1
    flags[inds] = False
    # Convention: multiply v by -1
    uvw[:, 1] *= -1

    # TODO Determine which observation is calibration observation
    # TODO Import name of source
    observations = []
    for ii in set(fieldid):
        mask = fieldid == ii
        if nspws > 1:
            mask = np.logical_and(mask, spwmask)
        myvis, myweight = vis[mask], weight[mask]
        myvis = np.ascontiguousarray(np.transpose(vis[mask], (2, 0, 1)))
        myweight = np.ascontiguousarray(np.transpose(weight[mask], (2, 0, 1)))
        myflags = np.ascontiguousarray(np.transpose(flags[mask], (2, 0, 1)))
        observations.append(Observation(antpos[mask], myvis, myweight, myflags,
                                        polarization, freqs, dirs[ii]))
    return observations
