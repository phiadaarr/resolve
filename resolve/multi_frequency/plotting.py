# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import matplotlib.pyplot as plt
import numpy as np

import nifty7 as ift
import os

from ..util import my_asserteq


def mf_plot(name, fld, movie_length=0):
    fdom, dom = fld.domain
    freqs = np.array(fdom.coordinates)
    nfreq = fld.shape[0]
    my_asserteq(nfreq, len(freqs))
    mi, ma = np.min(fld.val), np.max(fld.val)

    N = 10
    inds = np.random.choice(np.arange(dom.size), N)
    val = fld.val.reshape(fld.shape[0], -1).T
    plt.figure()
    for ind in inds:
        plt.plot(freqs*1e-6, val[ind])
    plt.xlabel('MHz')
    plt.tight_layout()
    plt.savefig(f'{name}_spectra.png')
    plt.close()

    if movie_length is None:
        return

    for ii in range(nfreq):
        ift.single_plot(ift.makeField(dom, fld.val[ii]),
                        title=f'{freqs[ii]*1e-6:.0f} MHz',
                        vmin=mi, vmax=ma,
                        cmap='inferno',
                        name=f'{name}_{ii:04.0f}.png')

    if movie_length > 0:
        framerate = nfreq/float(movie_length)
        os.system(f"ffmpeg -framerate {framerate} -i {name}_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 23 -y {name}.mp4")
        for ii in range(nfreq):
            os.remove(f'{name}_{ii:04.0f}.png')

