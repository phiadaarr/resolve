# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import argparse

import resolve as rve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ms")
    args = parser.parse_args()
    nspec = rve.ms_n_spectral_windows(args.ms)
    from casacore.tables import tablesummary
    tablesummary(args.ms)
    print()
    print(f"The data set has {nspec} spectral windows.")
    print()
