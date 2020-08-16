# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import sys

import resolve as rve


def main():
    _, ms = sys.argv
    obs = rve.ms2observations(ms, 'DATA')
    print(f'{len(obs)} fields.')
    for oo in obs:
        print(f'vis.shape {oo.vis.shape}')
        oo = obs.restrict_to_stokes_i()
        print(f'Stokes I only: vis.shape {oo.vis.shape}')
        print(f'Max snr {oo.max_snr()}')
        print(f'Fraction flagged {oo.fraction_flagged()}')
        print()


if __name__ == '__main__':
    main()
