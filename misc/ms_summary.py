#!/usr/bin/python3
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
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
