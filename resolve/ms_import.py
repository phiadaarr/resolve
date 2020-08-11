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

    with table(join(ms, 'SPECTRAL_WINDOW'), **CFG) as t:
        freqs = t.getcol('CHAN_FREQ')
    with table(join(ms, 'POLARIZATION'), **CFG) as t:
        polarization = t.getcol('CORR_TYPE')
        my_asserteq(polarization.ndim, 2)
        my_asserteq(polarization.shape[0], 1)
        polarization = Polarization(polarization[0])
    with table(join(ms, 'FIELD'), **CFG) as t:
        equinox = t.coldesc('REFERENCE_DIR')['desc']['keywords']['MEASINFO']['Ref']
        equinox = str(equinox)[1:]
        if equinox == "1950_VLA":
            equinox = 1950
        dirs = []
        for pc in t.getcol('REFERENCE_DIR'):
            my_asserteq(pc.shape, (1, 2))
            dirs.append(Direction(pc[0], equinox))
        dirs = tuple(dirs)
    with table(ms, **CFG) as t:
        vis = t.getcol(data_column)
        print(f'vis data type is {vis.dtype}')
        uvw = t.getcol('UVW')
        col = 'WEIGHT_SPECTRUM' if 'WEIGHT_SPECTRUM' in t.colnames() else 'WEIGHT'
        weight = t.getcol(col)
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
        weight = weight[:, None]
    weight[flags] = 0

    observations = []
    for ii in set(fieldid):
        mask = fieldid == ii
        if nspws > 1:
            mask = np.logical_and(mask, spwmask)
        myvis, myweight = vis[mask], weight[mask]
        myvis = np.ascontiguousarray(np.transpose(vis[mask], (2, 0, 1)))
        myweight = np.ascontiguousarray(np.transpose(weight[mask], (2, 0, 1)))
        observations.append(Observation(antpos[mask], myvis, myweight,
                                        polarization, freqs, dirs[ii]))
    return observations
