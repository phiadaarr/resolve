# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import sys

import resolve as rve


def main():
    _, ms, base = sys.argv
    obs = rve.ms2observations(ms, 'DATA')
    for ii, oo in enumerate(obs):
        oo.save_to_hdf5(f'{base}_{ii}.hdf5')


if __name__ == '__main__':
    main()
