# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse

import resolve as rve
from os.path import splitext


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz")
    args = parser.parse_args()

    # obs = rve.ms2observations(args.npz, "DATA", False, 0, "stokesiavg")[0]

    obs = rve.Observation.load(args.npz)

    for ii in range(len(obs.freq)):
        print(ii, len(obs.freq))
        rve.Observation(
            obs.antenna_positions,
            obs.vis.val[..., ii : ii + 1],
            obs.weight.val[..., ii : ii + 1],
            obs.polarization,
            obs.freq[ii : ii + 1],
            obs.direction,
        ).save(splitext(args.npz)[0] + "_ch" + str(ii) + ".npz", True)


if __name__ == "__main__":
    main()
