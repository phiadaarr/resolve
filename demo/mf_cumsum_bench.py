# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np
import matplotlib.pyplot as plt


def bench_cumsum_helper(npix, nfreq):
    from time import time
    a = np.zeros((2, nfreq, npix, npix))
    t0 = time()
    np.cumsum(a[0], axis=0)
    np.cumsum(a[1], axis=0)
    t1 = time()

    s0 = time()
    b = np.transpose(a, (0, 2, 3, 1))
    b = np.ascontiguousarray(b)
    np.cumsum(b[0], axis=-1)
    np.cumsum(b[1], axis=-1)
    b = np.transpose(a, (0, 3, 1, 2))
    b = np.ascontiguousarray(b)
    s1 = time()

    return t1-t0, s1-s0


def bench_cumsum():
    xs = [64, 70, 80, 93, 100, 128, 150, 189, 200, 256, 500, 512, 995, 999,
          1000, 1022, 1024, 2000, 2048]
    ys0 = []
    ys1 = []
    for ii, npix in enumerate(xs):
        print(ii, len(xs))
        nai, transp = bench_cumsum_helper(npix, 40)
        ys0.append(nai)
        ys1.append(transp)
    plt.plot(xs, ys0, label='Naive')
    plt.plot(xs, ys1, label='With transpose')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('bench_cumsum.png')
    print('Wrote bench_cumsum.png')


if __name__ == '__main__':
    bench_cumsum()
