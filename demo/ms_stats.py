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
    print(f'{len(obs)} fields.')
    for oo in obs:
        print(f'vis.shape {oo.vis.shape} ({oo.fraction_flagged()} flagged)')
        oo = oo.restrict_to_stokes_i()
        print(f'Stokes I only: vis.shape {oo.vis.shape} ({oo.fraction_flagged()} flagged)')
        oo = oo.average_stokes_i()
        print(f'Averaged Stokes I: vis.shape {oo.vis.shape} ({oo.fraction_flagged()} flagged)')
        print(f'Max snr {oo.max_snr()}')
        print()


if __name__ == '__main__':
    main()
