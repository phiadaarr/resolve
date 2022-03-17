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
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift

from .data.observation import Observation
from .mpi import master


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
