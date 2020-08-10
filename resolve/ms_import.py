# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

from os.path import isdir, join, splitext

import numpy as np

from .direction import Direction
from .observation import Observation
from .polarization import Polarization
from .util import my_assert


def ms2observations(ms, data_column, spectral_window=None):
    from casacore.tables import table
    CFG = {'readonly': True, 'ack': False}
    if not isdir(ms) or splitext(ms)[1] != '.ms':
        raise RuntimeError

    with table(join(ms, 'SPECTRAL_WINDOW'), **CFG) as t:
        freqs = t.getcol('CHAN_FREQ')
    with table(join(ms, 'POLARIZATION'), **CFG) as t:
        polarization = t.getcol('CORR_TYPE')
        my_assert(polarization.ndim == 2)
        my_assert(polarization.shape[0] == 1)
        polarization = Polarization(polarization[0])
    with table(join(ms, 'FIELD'), **CFG) as t:
        equ = t.coldesc('REFERENCE_DIR')['desc']['keywords']['MEASINFO']['Ref']
        dirs = []
        for pc in t.getcol('REFERENCE_DIR'):
            my_assert(pc.shape == (1, 2))
            dirs.append(Direction(pc[0], equ))
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

    nspws = len(np.unique(spw))
    if nspws > 1:
        if spectral_window is None:
            raise RuntimeError
        spwmask = spw == spectral_window
        freqs = freqs[spectral_window]
    else:
        freqs = freqs[0]

    observations = []
    for ii in set(fieldid):
        mask = fieldid == ii
        if nspws > 1:
            mask = np.logical_and(mask, spwmask)
        observations.append(Observation(uvw[mask], vis[mask], weight[mask],
                                        flags[mask], polarization, freqs,
                                        dirs[ii]))
    return observations
