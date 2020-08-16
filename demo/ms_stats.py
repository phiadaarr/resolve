# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import sys

import resolve as rve


def main():
    _, ms = sys.argv
    if ms[:-3] == '.ms':
        obs = rve.ms2observations(ms, 'DATA')
    else:
        obs = [rve.Observation.load_from_hdf5(ms)]
    for oo in obs:
        print(f'vis.shape {oo.vis.shape} ({oo.fraction_useful()*100:.2f}% useful)')
        print(f'Max snr {oo.max_snr()}')
        oo = oo.restrict_to_stokes_i()
        print('Stokes I only')
        print(f'vis.shape {oo.vis.shape} ({oo.fraction_useful()*100:.2f}% useful)')
        print(f'Max snr {oo.max_snr()}')
        oo = oo.average_stokes_i()
        print('Averaged Stokes I')
        print(f'vis.shape {oo.vis.shape} ({oo.fraction_useful()*100:.2f}% useful)')
        print(f'Max snr {oo.max_snr()}')
        print()


if __name__ == '__main__':
    main()
