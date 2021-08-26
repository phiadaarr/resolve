#!/usr/bin/python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import argparse
from os.path import join

import resolve as rve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ms")
    args = parser.parse_args()
    nspec = rve.ms_n_spectral_windows(args.ms)
    from casacore.tables import tablesummary
    tablesummary(args.ms)

    with rve.ms_table(join(args.ms, "SPECTRAL_WINDOW")) as t:
        for ii in range(nspec):
            print(f"Spectral window #{ii}")
            chans = t.getcol("CHAN_FREQ", startrow=ii, nrow=1)

            print(f"Shape: {chans.shape}")
            print(f"f1-f0: {(chans[0][1]-chans[0][0])/1e6} MHz")
            print("Frequencies (GHz)")
            print(chans/1e9)
            print()

    with rve.ms_table(join(args.ms, "FIELD")) as t:
        name = t.getcol("NAME")
        refdir = t.getcol("REFERENCE_DIR")
        deldir = t.getcol("DELAY_DIR")
        phdir = t.getcol("PHASE_DIR")

    print("NAME REFERNCE_DIR DELAY_DIR PHASE_DIR")
    for nn, rd, dd, pd in zip(name, refdir, deldir, phdir):
        print(nn, rd, dd, pd)

    with rve.ms_table(join(args.ms, "POINTING")) as t:
        print(t.getcol("ANTENNA_ID"))
        print(t.getcol("TIME"))
        print(t.getcol("DIRECTION").shape)
