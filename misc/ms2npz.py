#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import argparse
from os.path import join, split, splitext
from os import makedirs

import resolve as rve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-column", default="DATA")
    parser.add_argument("--output-folder", default=".")
    parser.add_argument("--include-calibration-info", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--spectral-window", default=0, type=int)
    parser.add_argument("--ch-begin", type=int)
    parser.add_argument("--ch-end", type=int)
    parser.add_argument("--ch-jump", type=int)
    parser.add_argument("--ignore-flags", action="store_true")
    parser.add_argument("--autocorrelations-only", action="store_true",
                        help=("If this flag is set, all autocorrelations are "
                              "imported irrespective of whether they are "
                              "flagged or not."))
    parser.add_argument("ms")
    parser.add_argument(
        "polarization_mode",
        help="Can be 'stokesiavg', 'stokesi', 'all', or something like 'LL' or 'XY'.",
    )
    args = parser.parse_args()
    makedirs(args.output_folder, exist_ok=True)
    name = splitext(split(args.ms)[1])[0]
    nspec = rve.ms_n_spectral_windows(args.ms)
    print(f"The data set has {nspec} spectral windows. Select {args.spectral_window}.")
    obs = rve.ms2observations(
        args.ms,
        args.data_column,
        args.include_calibration_info,
        args.spectral_window,
        args.polarization_mode,
        slice(args.ch_begin, args.ch_end, args.ch_jump),
        args.ignore_flags
    )
    for ifield, oo in enumerate(obs):
        if args.autocorrelations_only:
            oo = oo.restrict_to_autocorrelations()
            auto = "autocorrelationsonly"
        else:
            auto = ""
        fname = join(args.output_folder, f"{name}field{ifield}{args.data_column}{auto}.npz")
        print(f"Save {fname}")
        oo.save(fname, args.compress)
        if oo.vis.size == 0:
            print(f"WARNING: {fname} is empty")
