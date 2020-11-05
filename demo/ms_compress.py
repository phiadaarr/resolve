# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import sys
import resolve as rve


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise RuntimeError("bad number of command line arguments")
    # Parameters
    datacolumn = "DATA"
    with_calib_info = True
    datasetname = sys.argv[1]
    outname = sys.argv[2]
    nspec = rve.ms_n_spectral_windows(datasetname)

    for compr in [True, False]:
        for pol in ["LL", "stokesiavg", "stokesi", "all"]:
            for spec in range(nspec):
                obs = rve.ms2observations(datasetname, datacolumn, with_calib_info, spec, pol)
                for ifield, oo in enumerate(obs):
                    fname = f"{outname}compr{compr}pol{pol}spectralwindow{spec}field{ifield}.npz"
                    print(f"Save {fname}")
                    oo.save_to_npz(fname, compr)
