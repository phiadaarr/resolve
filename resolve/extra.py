# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift

from .mpi import master
from data.observation import Observation


def split_data_file(data_path, n_task, target_folder, base_name, n_work, compress):
    from os import makedirs
    makedirs(target_folder, exist_ok=True)

    obs = Observation.load(data_path)

    for rank in range(n_task):
        lo, hi = ift.utilities.shareRange(n_work, n_task, rank)
        sliced_obs = obs.get_freqs_by_slice(slice(*(lo, hi)))
        sliced_obs.save(f"{target_folder}/{base_name}_{rank}.npz", compress=compress)


def mpi_load(data_folder, base_name, full_data_set, n_work, comm=None, compress=False):
    if master:
        from os.path import isdir
        if not isdir(data_folder):
            split_data_file(full_data_set, comm.Get_size(), data_folder, base_name, n_work, compress)
        if comm is None:
            return Observation.load(full_data_set)

    comm.Barrier()
    return Observation.load(f"{data_folder}/{base_name}_{comm.Get_rank()}.npz")
